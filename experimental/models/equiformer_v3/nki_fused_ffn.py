"""
Fused FFN NKI Kernel for EquiformerV3
=====================================

Fuses: to_grid + GatedSwiGLUGridMLP + from_grid into a single NKI kernel,
eliminating the 12.2 GB DMA spill from the [N, 400, 128] grid intermediate.

Computation per atom n:
    1. to_grid:       grid = to_grid_mat[400,25] @ x_n[25,128]       → [400, 128]
    2. grid_linear_1: grid = grid @ W1.T                             → [400, 256]
    3. chunk + gate:  grid = grid_1 * sigmoid(scalar @ Wg.T + bg) * grid_2  → [400, 128]
    4. grid_linear_2: grid = grid @ W2.T                             → [400, 128]
    5. from_grid:     y_n = from_grid_mat_T[400,25].T @ grid[400,128] → [25, 128]

Data flow through SBUF (no HBM intermediates!):
    to_grid → [C=128, grid_tile=100] → grid_linear_1 → [grid_tile=100, 256]
    → chunk+gate → [grid_tile=100, 128] → transpose → [C=128, grid_tile=100]
    → grid_linear_2 → [grid_tile=100, 128] → from_grid (accumulate in PSUM)
    → [25, 128] → store

Pre-transposed weight layouts (caller responsibility):
    - to_grid_mat:     [25, 400]  — original layout from SO3Grid
    - from_grid_mat_T: [400, 25]  — caller does from_grid_mat.T.contiguous()
    - W1:              [128, 256] — caller does grid_linear_1.weight.T.contiguous()
    - W2:              [128, 128] — caller does grid_linear_2.weight.T.contiguous()
    - W_gate:          [128, 128] — caller does gating_linear.weight.T.contiguous()
    - b_gate:          [1, 128]   — gating_linear.bias.unsqueeze(0)

nc_matmul semantics:
    nc_matmul(dst, stationary[K,M], moving[K,N]) = stat^T @ mov = [M,N]
    K = partition (contraction, ≤128), M = stat_free (≤128), N = mov_free (≤512)
    Result [M, N]: M = PSUM partition, N = PSUM free

Tiling:
    - Grid dim (400) → 4 tiles of 100
    - C=128 = full partition for grid_MLP matmuls
    - j=25 as partition for to_grid/from_grid (20% utilization, unavoidable)
    - Atoms processed sequentially (outer loop)

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


@nki.jit
def fused_ffn_grid_kernel(
    x_hbm,  # [N*25, 128] input features
    scalars_hbm,  # [N, 128] scalar features for gating
    to_grid_mat_hbm,  # [25, 400]
    from_grid_mat_T_hbm,  # [400, 25] (transposed by caller)
    w1_hbm,  # [128, 256] grid_linear_1.weight.T
    w2_hbm,  # [128, 128] grid_linear_2.weight.T
    w_gate_hbm,  # [128, 128] gating_linear.weight.T
    b_gate_hbm,  # [1, 128] gating_linear.bias
):
    """
    Fused to_grid + GatedSwiGLUGridMLP + from_grid NKI kernel.

    Eliminates all [N, 400, 128] intermediate spills by keeping grid data
    in SBUF throughout the computation pipeline.

    Returns:
        y_hbm: [N*25, 128] output features
    """
    N_times_SH = x_hbm.shape[0]
    N = N_times_SH // SH_DIM

    # Allocate output
    y_hbm = nl.ndarray((N_times_SH, C_DIM), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    # =========================================================================
    # Load constant weights (persist in SBUF across all atoms)
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
    # Per-atom computation
    # =========================================================================
    for n in nl.sequential_range(N):
        # --- Compute gate: sigmoid(scalar_n @ W_gate.T + b_gate) → [1, C=128] ---
        # Load scalar_n [1, 128] from HBM
        scalar_n = nl.ndarray((1, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=scalar_n[0:1, 0:C_DIM], src=scalars_hbm[n : n + 1, 0:C_DIM])

        # Transpose scalar to column: [1, 128] → [128, 1] for matmul
        scalar_col_psum = nl.ndarray((C_DIM, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=scalar_col_psum, data=scalar_n)
        scalar_col = nl.ndarray((C_DIM, 1), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=scalar_col, src=scalar_col_psum)

        # gate = scalar_col^T @ w_gate = scalar_n @ W_gate.T → [1, 128]
        # nc_matmul(dst, stat=[128,1], mov=[128,128]) → [1, 128]
        gate_psum = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=gate_psum, stationary=scalar_col, moving=w_gate_sbuf)

        # PSUM → SBUF, add bias, apply sigmoid
        gate_f32 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gate_f32, src=gate_psum)

        # Add bias (cast bias to f32 for addition)
        b_gate_f32 = nl.ndarray((1, C_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=b_gate_f32, data=b_gate_sbuf, op=nl.copy)
        nisa.tensor_tensor(dst=gate_f32, data1=gate_f32, data2=b_gate_f32, op=nl.add)

        # Sigmoid
        nisa.activation(dst=gate_f32, data=gate_f32, op=nl.sigmoid)
        # gate_f32: [1, 128] — the gating values (P=1, free=128)

        # --- Load x_n [25, 128] from HBM (P=25, free=128) ---
        x_n = nl.ndarray((SH_DIM, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_n[0:SH_DIM, 0:C_DIM],
            src=x_hbm[n * SH_DIM : (n + 1) * SH_DIM, 0:C_DIM],
        )

        # --- from_grid accumulator (accumulates across 4 grid tiles) ---
        # Result shape: [25, 128] in PSUM (P=25, free=128)
        from_grid_psum = nl.ndarray((SH_DIM, C_DIM), dtype=nl.float32, buffer=nl.psum)

        # --- Process each grid tile ---
        for gt in nl.affine_range(NUM_GRID_TILES):
            # ===============================================================
            # STAGE 1: to_grid
            # nc_matmul(dst, stat=x_n[P=25, M=128], mov=to_grid_tile[P=25, N=100])
            # = x_n^T @ to_grid_tile = [M=128, N=100]
            # Result in PSUM: [128, 100] with P_out=128, F_out=100
            # ===============================================================
            to_grid_tile = nl.ndarray(
                (SH_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=to_grid_tile[0:SH_DIM, 0:GRID_TILE],
                src=to_grid_mat_hbm[0:SH_DIM, gt * GRID_TILE : (gt + 1) * GRID_TILE],
            )

            to_grid_psum = nl.ndarray(
                (C_DIM, GRID_TILE), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(dst=to_grid_psum, stationary=x_n, moving=to_grid_tile)
            # Result: [C=128, grid_tile=100], P=C=128

            # Copy to SBUF: grid_in [128, 100]
            grid_in = nl.ndarray((C_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=grid_in, src=to_grid_psum)

            # ===============================================================
            # STAGE 2: grid_linear_1
            # grid_in is [P=128, F=100]. We want: grid_in.T @ W1 = [100, 128] @ [128, 256] → [100, 256]
            # nc_matmul(dst, stat=grid_in[P=128, M=100], mov=W1[P=128, N=256])
            # = grid_in^T @ W1 = [100, 256]
            # Result: [M=100, N=256] in PSUM, P_out=100, F_out=256
            # ===============================================================
            gl1_psum = nl.ndarray(
                (GRID_TILE, 2 * C_DIM), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_matmul(dst=gl1_psum, stationary=grid_in, moving=w1_sbuf)

            # Copy to SBUF: [100, 256] with P=100
            gl1_sbuf = nl.ndarray(
                (GRID_TILE, 2 * C_DIM), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=gl1_sbuf, src=gl1_psum)

            # ===============================================================
            # STAGE 3: Chunk + Gate + SwiGLU
            # gl1_sbuf [P=100, F=256]: chunk into grid_1[100,128], grid_2[100,128]
            # gate_f32 [P=1, F=128]: broadcast to [P=100, F=128]
            # output = grid_1 * gate * grid_2 → [100, 128]
            # ===============================================================

            # Chunk: first and second halves along free dimension
            # grid_1 = gl1_sbuf[:, 0:128], grid_2 = gl1_sbuf[:, 128:256]

            # Broadcast gate from [1, 128] to [100, 128]
            gate_bc = nl.broadcast_to(gate_f32, (GRID_TILE, C_DIM))

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
            # swiglu_out: [P=100, F=128]

            # ===============================================================
            # STAGE 4: grid_linear_2
            # Need: swiglu_out[100, 128] @ W2[128, 128].T → [100, 128]
            # = swiglu_out[100, 128] @ W2[128, 128] → [100, 128]
            #
            # But swiglu_out has P=100 and W2 has P=128 — partition mismatch!
            # Solution: transpose swiglu_out from [P=100, F=128] to [P=128, F=100]
            # Then: nc_matmul(dst, stat=swiglu_T[P=128, M=100], mov=W2[P=128, N=128])
            # = swiglu_T^T @ W2 = swiglu_out @ W2 = [100, 128]
            # ===============================================================

            # Transpose: [100, 128] → [128, 100]
            swiglu_T_psum = nl.ndarray(
                (C_DIM, GRID_TILE), dtype=nl.float32, buffer=nl.psum
            )
            nisa.nc_transpose(dst=swiglu_T_psum, data=swiglu_out)
            swiglu_T = nl.ndarray((C_DIM, GRID_TILE), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=swiglu_T, src=swiglu_T_psum)

            # grid_linear_2 matmul
            gl2_psum = nl.ndarray((GRID_TILE, C_DIM), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=gl2_psum, stationary=swiglu_T, moving=w2_sbuf)
            # Result: [M=100, N=128] in PSUM, P_out=100

            # Copy to SBUF: processed_grid [100, 128] with P=100
            processed_grid = nl.ndarray(
                (GRID_TILE, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=processed_grid, src=gl2_psum)

            # ===============================================================
            # STAGE 5: from_grid (accumulates across grid tiles)
            # from_grid: y[j,c] = sum_a from_grid_mat[j,a] * grid[a,c]
            # from_grid_mat_T is [400, 25] → tile to [grid_tile=100, 25]
            #
            # nc_matmul(dst, stat=fg_tile[P=100, M=25], mov=processed_grid[P=100, N=128])
            # = fg_tile^T @ processed_grid = [25, 128]
            # P_in = 100 (contraction = grid points)
            # Result: [M=25, N=128] in PSUM (P_out=25, F_out=128)
            #
            # Writing to same from_grid_psum across 4 iterations triggers
            # hardware accumulation (K-tiling over grid dimension).
            # ===============================================================
            fg_tile = nl.ndarray((GRID_TILE, SH_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=fg_tile[0:GRID_TILE, 0:SH_DIM],
                src=from_grid_mat_T_hbm[
                    gt * GRID_TILE : (gt + 1) * GRID_TILE, 0:SH_DIM
                ],
            )

            nisa.nc_matmul(
                dst=from_grid_psum, stationary=fg_tile, moving=processed_grid
            )

        # --- Store output: from_grid_psum [25, 128] → y_hbm ---
        y_n = nl.ndarray((SH_DIM, C_DIM), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=y_n, src=from_grid_psum)
        nisa.dma_copy(
            dst=y_hbm[n * SH_DIM : (n + 1) * SH_DIM, 0:C_DIM],
            src=y_n[0:SH_DIM, 0:C_DIM],
        )

    return y_hbm
