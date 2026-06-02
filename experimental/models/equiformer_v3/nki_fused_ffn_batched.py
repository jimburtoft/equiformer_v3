"""
Batched Fused FFN NKI Kernel for EquiformerV3
==============================================

Same fusion as nki_fused_ffn.py (to_grid + GatedSwiGLUGridMLP + from_grid)
but processes ATOM_TILE=4 atoms per unrolled iteration, reducing the
unroll count from N to N/4 and enabling parallel execution within each batch.

Key difference from per-atom kernel:
    - Outer loop: nl.static_range(N // ATOM_TILE) — 25 iterations at N=100 (vs 100)
    - Inner loop: nl.affine_range(ATOM_TILE) — 4 parallel atom operations
    - to_grid uses concatenated atoms along free dim: [25, 4*128=512] → [100, 512]
    - grid_linear_1/2 process per-atom slices (128-wide) sequentially within the tile

Computation per batch of ATOM_TILE atoms:
    For each grid tile (4 tiles of 100):
        1. to_grid (batched):  stat=tg[25,100], mov=x_concat[25,512] → [100, 512]
        2. Per atom b in [0..3]:
           a. Extract + transpose grid_b[100,128] → [128,100]
           b. grid_linear_1: stat=grid_b_T[128,100], mov=W1[128,256] → [100, 256]
           c. SwiGLU: chunk + gate → [100, 128]
           d. Transpose [100,128] → [128,100]
           e. grid_linear_2: stat=swiglu_T[128,100], mov=W2[128,128] → [100, 128]
           f. from_grid: stat=fg[100,25], mov=processed[100,128] → accumulate [25, 128]
    Store 4 output atoms [4, 25, 128]

Hardware constraints verified:
    - to_grid: stat[25,100] M=100≤128 ✓, mov[25,512] N=512≤512 ✓
    - grid_linear_1: stat[128,100] M=100≤128 ✓, mov[128,256] N=256≤512 ✓
    - grid_linear_2: stat[128,100] M=100≤128 ✓, mov[128,128] N=128≤512 ✓
    - from_grid: stat[100,25] M=25≤128 ✓, mov[100,128] N=128≤512 ✓
    - PSUM: max free dim = 512 (to_grid result [100,512]) ✓

Requires: N must be divisible by ATOM_TILE (caller pads if needed).

Tested on: NKI 0.4.0, trn2.3xlarge (gen3)
"""

import nki
import nki.isa as nisa
import nki.language as nl

# Hardware constants
P_MAX = 128  # Max partition dimension
PSUM_FMAX = 512  # Max PSUM free dimension (gen3)
GRID_TILE = 100  # Grid dimension tile size (400 / 4)
NUM_GRID_TILES = 4  # Number of grid tiles
GRID_DIM = 400  # Total grid points
SH_DIM = 25  # Spherical harmonic coefficients ((lmax+1)^2)
C_DIM = 128  # Channel dimension
ATOM_TILE = 4  # Atoms processed per unrolled iteration


@nki.jit
def fused_ffn_grid_batched_kernel(
    x_hbm,  # [N*25, 128] input features (N must be divisible by ATOM_TILE)
    scalars_hbm,  # [N, 128] scalar features for gating
    to_grid_mat_hbm,  # [25, 400]
    from_grid_mat_T_hbm,  # [400, 25] (transposed by caller)
    w1_hbm,  # [128, 256] grid_linear_1.weight.T
    w2_hbm,  # [128, 128] grid_linear_2.weight.T
    w_gate_hbm,  # [128, 128] gating_linear.weight.T
    b_gate_hbm,  # [1, 128] gating_linear.bias
):
    """
    Batched fused to_grid + GatedSwiGLUGridMLP + from_grid NKI kernel.

    Processes ATOM_TILE=4 atoms per iteration, reducing unroll count by 4x.
    N must be divisible by ATOM_TILE.

    Returns:
        y_hbm: [N*25, 128] output features
    """
    N_times_SH = x_hbm.shape[0]
    N = N_times_SH // SH_DIM
    N_BATCHES = N // ATOM_TILE

    # Allocate output
    y_hbm = nl.ndarray((N_times_SH, C_DIM), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    # =========================================================================
    # Load constant weights (persist in SBUF across all iterations)
    # =========================================================================

    # W1: grid_linear_1.weight.T [128, 256]
    w1_sbuf = nl.ndarray((C_DIM, 2 * C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w1_sbuf[0:C_DIM, 0 : 2 * C_DIM], src=w1_hbm[0:C_DIM, 0 : 2 * C_DIM]
    )

    # W2: grid_linear_2.weight.T [128, 128]
    w2_sbuf = nl.ndarray((C_DIM, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2_sbuf[0:C_DIM, 0:C_DIM], src=w2_hbm[0:C_DIM, 0:C_DIM])

    # W_gate: gating_linear.weight.T [128, 128]
    w_gate_sbuf = nl.ndarray((C_DIM, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_gate_sbuf[0:C_DIM, 0:C_DIM], src=w_gate_hbm[0:C_DIM, 0:C_DIM])

    # b_gate: gating_linear.bias [1, 128]
    b_gate_sbuf = nl.ndarray((1, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_gate_sbuf[0:1, 0:C_DIM], src=b_gate_hbm[0:1, 0:C_DIM])

    # =========================================================================
    # Process atoms in batches of ATOM_TILE
    # =========================================================================
    for batch_idx in nl.static_range(N_BATCHES):
        base_atom = batch_idx * ATOM_TILE

        # --- Compute gates for all ATOM_TILE atoms in this batch ---
        # gate_b[b] = sigmoid(scalar_b @ W_gate.T + b_gate) → [1, 128] per atom
        # Use separate [1, 128] buffers to avoid [4, 128] allocation issues
        gate_0 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
        gate_1 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
        gate_2 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
        gate_3 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
        gates = [gate_0, gate_1, gate_2, gate_3]

        for b in nl.affine_range(ATOM_TILE):
            atom_idx = base_atom + b
            # Load scalar [1, 128]
            scalar_b = nl.ndarray((1, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=scalar_b[0:1, 0:C_DIM],
                src=scalars_hbm[atom_idx : atom_idx + 1, 0:C_DIM],
            )

            # Transpose: [1, 128] → [128, 1]
            # On gen3+, nc_transpose dst dtype must match input dtype
            scalar_col_psum = nl.ndarray((C_DIM, 1), dtype=x_hbm.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=scalar_col_psum, data=scalar_b)
            scalar_col = nl.ndarray((C_DIM, 1), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=scalar_col, src=scalar_col_psum)

            # gate = scalar @ W_gate.T: stat[128,1], mov[128,128] → [1, 128]
            gate_psum = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=gate_psum, stationary=scalar_col, moving=w_gate_sbuf)

            # Copy to SBUF, add bias, sigmoid
            gate_tmp = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=gate_tmp, src=gate_psum)

            b_gate_f32 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=b_gate_f32, data=b_gate_sbuf, op=nl.copy)
            nisa.tensor_tensor(
                dst=gate_tmp, data1=gate_tmp, data2=b_gate_f32, op=nl.add
            )
            nisa.activation(dst=gate_tmp, data=gate_tmp, op=nl.sigmoid)

            # Store in per-atom gate buffer
            nisa.tensor_copy(dst=gates[b], src=gate_tmp)

        # --- Load x for all ATOM_TILE atoms: x_batch[ATOM_TILE*25, 128] ---
        # Reshape to [25, ATOM_TILE*128] = [25, 512] for batched to_grid
        x_concat = nl.ndarray(
            (SH_DIM, ATOM_TILE * C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf
        )
        for b in nl.affine_range(ATOM_TILE):
            atom_idx = base_atom + b
            # Load x_b[25, 128] into the b-th 128-wide slice of x_concat
            nisa.dma_copy(
                dst=x_concat[0:SH_DIM, b * C_DIM : (b + 1) * C_DIM],
                src=x_hbm[atom_idx * SH_DIM : (atom_idx + 1) * SH_DIM, 0:C_DIM],
            )

        # --- from_grid accumulators: one per atom [25, 128] ---
        # Use separate 2D PSUM buffers per atom (PSUM is always 2D: [P, F])
        fg_psum_0 = nl.ndarray((SH_DIM, C_DIM), dtype=nl.float32, buffer=nl.psum)
        fg_psum_1 = nl.ndarray((SH_DIM, C_DIM), dtype=nl.float32, buffer=nl.psum)
        fg_psum_2 = nl.ndarray((SH_DIM, C_DIM), dtype=nl.float32, buffer=nl.psum)
        fg_psum_3 = nl.ndarray((SH_DIM, C_DIM), dtype=nl.float32, buffer=nl.psum)

        # --- Process each grid tile ---
        for gt in nl.affine_range(NUM_GRID_TILES):
            # Load to_grid tile [25, 100]
            tg_tile = nl.ndarray((SH_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tg_tile[0:SH_DIM, 0:GRID_TILE],
                src=to_grid_mat_hbm[0:SH_DIM, gt * GRID_TILE : (gt + 1) * GRID_TILE],
            )

            # =================================================================
            # STAGE 1: Batched to_grid
            # stat=tg_tile[P=25, M=100], mov=x_concat[P=25, N=512]
            # Result: [100, 512] in PSUM — 4 atoms interleaved
            # grid_concat[:, b*128:(b+1)*128] = grid for atom b
            # =================================================================
            grid_concat_psum = nl.ndarray(
                (GRID_TILE, ATOM_TILE * C_DIM), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(dst=grid_concat_psum, stationary=tg_tile, moving=x_concat)

            # Copy to SBUF: [100, 512]
            grid_concat = nl.ndarray(
                (GRID_TILE, ATOM_TILE * C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=grid_concat, src=grid_concat_psum)

            # Load from_grid tile [100, 25]
            fg_tile = nl.ndarray((GRID_TILE, SH_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=fg_tile[0:GRID_TILE, 0:SH_DIM],
                src=from_grid_mat_T_hbm[
                    gt * GRID_TILE : (gt + 1) * GRID_TILE, 0:SH_DIM
                ],
            )

            # =================================================================
            # STAGES 2-5: Per-atom grid MLP + from_grid
            # Process each atom's 128-wide slice through the MLP pipeline
            #
            # Use Python list for PSUM indexing since nl.affine_range unrolls
            # b to compile-time constants (0, 1, 2, 3).
            # =================================================================
            fg_psums = [fg_psum_0, fg_psum_1, fg_psum_2, fg_psum_3]

            for b in nl.affine_range(ATOM_TILE):
                # Extract atom b's grid data: grid_concat[:, b*128:(b+1)*128]
                # Shape: [P=100, F=128]
                grid_b = nl.ndarray(
                    (GRID_TILE, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(
                    dst=grid_b[0:GRID_TILE, 0:C_DIM],
                    src=grid_concat[0:GRID_TILE, b * C_DIM : (b + 1) * C_DIM],
                )

                # --- Stage 2: grid_linear_1 ---
                # Transpose grid_b[100, 128] → [128, 100]
                # On gen3+, nc_transpose dst dtype must match input dtype
                grid_b_T_psum = nl.ndarray(
                    (C_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.psum
                )
                nisa.nc_transpose(dst=grid_b_T_psum, data=grid_b)
                grid_b_T = nl.ndarray(
                    (C_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=grid_b_T, src=grid_b_T_psum)

                # grid_linear_1: stat=grid_b_T[128,100], mov=W1[128,256] → [100, 256]
                gl1_psum = nl.ndarray(
                    (GRID_TILE, 2 * C_DIM), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(dst=gl1_psum, stationary=grid_b_T, moving=w1_sbuf)

                gl1_sbuf = nl.ndarray(
                    (GRID_TILE, 2 * C_DIM), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=gl1_sbuf, src=gl1_psum)

                # --- Stage 3: SwiGLU ---
                # gate for atom b: broadcast [1, 128] → [100, 128]
                gate_bc = nl.broadcast_to(gates[b], (GRID_TILE, C_DIM))

                # grid_1 * gate
                gated = nl.ndarray((GRID_TILE, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=gated,
                    data1=gl1_sbuf[0:GRID_TILE, 0:C_DIM],
                    data2=gate_bc,
                    op=nl.multiply,
                )

                # gated * grid_2
                swiglu_out = nl.ndarray(
                    (GRID_TILE, C_DIM), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_tensor(
                    dst=swiglu_out,
                    data1=gated,
                    data2=gl1_sbuf[0:GRID_TILE, C_DIM : 2 * C_DIM],
                    op=nl.multiply,
                )

                # --- Stage 4: grid_linear_2 ---
                # Transpose swiglu[100,128] → [128,100]
                swiglu_T_psum = nl.ndarray(
                    (C_DIM, GRID_TILE), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_transpose(dst=swiglu_T_psum, data=swiglu_out)
                swiglu_T = nl.ndarray(
                    (C_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=swiglu_T, src=swiglu_T_psum)

                # grid_linear_2: stat=swiglu_T[128,100], mov=W2[128,128] → [100, 128]
                gl2_psum = nl.ndarray(
                    (GRID_TILE, C_DIM), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_matmul(dst=gl2_psum, stationary=swiglu_T, moving=w2_sbuf)

                processed = nl.ndarray(
                    (GRID_TILE, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=processed, src=gl2_psum)

                # --- Stage 5: from_grid (accumulate across grid tiles) ---
                # stat=fg_tile[100,25], mov=processed[100,128] → [25, 128]
                # Writing to same PSUM buffer across grid tile iterations
                # triggers hardware accumulation.
                nisa.nc_matmul(
                    dst=fg_psums[b],
                    stationary=fg_tile,
                    moving=processed,
                )

        # --- Store outputs for all ATOM_TILE atoms ---
        fg_psums_store = [fg_psum_0, fg_psum_1, fg_psum_2, fg_psum_3]
        for b in nl.affine_range(ATOM_TILE):
            atom_idx = base_atom + b
            y_b = nl.ndarray((SH_DIM, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y_b, src=fg_psums_store[b])
            nisa.dma_copy(
                dst=y_hbm[atom_idx * SH_DIM : (atom_idx + 1) * SH_DIM, 0:C_DIM],
                src=y_b[0:SH_DIM, 0:C_DIM],
            )

    return y_hbm
