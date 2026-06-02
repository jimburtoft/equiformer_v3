"""
Fused NKI kernel for EquiformerV3: merge + wigner rotate.

V4c - Per-edge nc_matmul approach:
  For each E-tile of 128 edges:
    1. Merge: compute msg[E_TILE, M*C] in SBUF (same as V4)
    2. Rotate: for each edge in the tile, do ONE nc_matmul:
         wigner[25, 25] @ msg[25, 128] -> output[25, 128]
       This replaces 625 scalar_tensor_tensor calls with 128 nc_matmul calls.

Total instructions per tile: merge(~150) + rotate(128 matmuls) = ~278
Total instructions: 71 tiles * 278 = ~19,700
Compared to V4b: 71 * 625 = 44,375 scalar_tensor_tensor ops.

Trade-off: P=25 (underutilization) but nc_matmul is much higher throughput per op.
"""

import nki
import nki.isa as nisa
import nki.language as nl


def div_ceil(n, d):
    return (n + d - 1) // d


@nki.jit
def nki_fused_merge_rotate(
    x_source,  # [E, M, C]
    x_target,  # [E, M, C]
    edge_weight_src,  # [E, M, C]
    edge_weight_tgt,  # [E, M, C]
    wigner,  # [E, M, M]
):
    """
    Fused merge + wigner rotate using per-edge nc_matmul.

    Computes:
      msg = x_source * ew_src + x_target * ew_tgt   (element-wise)
      output[e, :, :] = wigner[e, :, :] @ msg[e, :, :]  (per-edge matmul)

    Returns: output [E, M, C]
    """
    E = x_source.shape[0]
    M = x_source.shape[1]  # 25
    C = x_source.shape[2]  # 128

    # Tile parameters
    E_TILE = 128
    F_MERGE = M * C  # 3200

    # Output in HBM
    output = nl.ndarray((E, M, C), dtype=x_source.dtype, buffer=nl.shared_hbm)

    num_e_tiles = div_ceil(E, E_TILE)

    for tile_idx in nl.affine_range(num_e_tiles):
        row_start = tile_idx * E_TILE
        row_end = min(row_start + E_TILE, E)
        p_size = row_end - row_start

        # ===== Phase 1: Merge (P=128, F=3200) =====
        msg_buf = nl.ndarray((E_TILE, F_MERGE), dtype=x_source.dtype, buffer=nl.sbuf)
        buf_a = nl.ndarray((E_TILE, C), dtype=x_source.dtype, buffer=nl.sbuf)
        buf_b = nl.ndarray((E_TILE, C), dtype=x_source.dtype, buffer=nl.sbuf)

        for m in nl.affine_range(M):
            f_start = m * C
            f_end = f_start + C

            nisa.dma_copy(
                dst=buf_a[0:p_size, 0:C], src=x_source[row_start:row_end, m, 0:C]
            )
            nisa.dma_copy(
                dst=buf_b[0:p_size, 0:C], src=edge_weight_src[row_start:row_end, m, 0:C]
            )
            nisa.tensor_tensor(
                dst=msg_buf[0:p_size, f_start:f_end],
                data1=buf_a[0:p_size, 0:C],
                data2=buf_b[0:p_size, 0:C],
                op=nl.multiply,
            )

            nisa.dma_copy(
                dst=buf_a[0:p_size, 0:C], src=x_target[row_start:row_end, m, 0:C]
            )
            nisa.dma_copy(
                dst=buf_b[0:p_size, 0:C], src=edge_weight_tgt[row_start:row_end, m, 0:C]
            )
            nisa.tensor_tensor(
                dst=buf_a[0:p_size, 0:C],
                data1=buf_a[0:p_size, 0:C],
                data2=buf_b[0:p_size, 0:C],
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=msg_buf[0:p_size, f_start:f_end],
                data1=msg_buf[0:p_size, f_start:f_end],
                data2=buf_a[0:p_size, 0:C],
                op=nl.add,
            )

        # ===== Phase 2: Wigner Rotation via per-edge nc_matmul =====
        # For each edge e in [0, p_size):
        #   wigner_e = wigner[row_start+e, :, :] -> [M, M] = [25, 25]
        #   msg_e = msg_buf[e, :] reshaped to [M, C] = [25, 128]
        #   output_e = wigner_e @ msg_e -> [M, C] = [25, 128]
        #
        # nc_matmul: dst[K_p, N_f] = stationary[K_p, M_f] @ moving[M_p, N_f]
        #   stationary = wigner_e[M, M] (P=M=25, F=M=25) -- wigner matrix
        #   moving = msg_e[M, C] (P=M=25, F=C=128) -- message
        #   result = [M, C] = [25, 128] in PSUM

        # Buffers for per-edge rotation (P=M=25)
        wigner_e = nl.ndarray((M, M), dtype=wigner.dtype, buffer=nl.sbuf)
        msg_e = nl.ndarray((M, C), dtype=x_source.dtype, buffer=nl.sbuf)
        out_e = nl.ndarray((M, C), dtype=x_source.dtype, buffer=nl.sbuf)

        for e in nl.affine_range(p_size):
            # Load wigner matrix for this edge: wigner[row_start+e, :, :] -> [M, M]
            # wigner is [E, M, M]. We need to load [1, M, M] -> [M, M]
            # Use 2D access: wigner[row_start+e, m_out, :] for each m_out
            for m_out in nl.affine_range(M):
                nisa.dma_copy(
                    dst=wigner_e[m_out, 0:M], src=wigner[row_start + e, m_out, 0:M]
                )

            # Extract msg for this edge from msg_buf[e, :] (P=E_TILE, F=M*C)
            # Need to reshape: msg_buf[e, :] is [1, M*C] -> [M, C]
            # Load from msg_buf partition e: msg_buf[e, m*C:(m+1)*C] for each m
            for m_in in nl.affine_range(M):
                f_start_in = m_in * C
                # Copy from msg_buf (P=E_TILE) at partition e to msg_e (P=M) at partition m_in
                # This is a cross-partition copy which isn't straightforward...
                # Actually msg_buf[e:e+1, f_start_in:f_start_in+C] is [1, C]
                # and msg_e[m_in, 0:C] is also [1, C] (1 partition)
                nisa.tensor_copy(
                    dst=msg_e[m_in, 0:C], src=msg_buf[e, f_start_in : f_start_in + C]
                )

            # nc_matmul: wigner_e[M, M] @ msg_e[M, C] -> [M, C]
            result_psum = nl.ndarray((M, C), dtype=nl.float32, buffer=nl.psum)
            result_psum[0:M, 0:C] = nisa.nc_matmul(
                stationary=wigner_e[0:M, 0:M], moving=msg_e[0:M, 0:C]
            )

            # Copy from PSUM to SBUF
            nisa.tensor_copy(dst=out_e[0:M, 0:C], src=result_psum[0:M, 0:C])

            # Store result: output[row_start+e, :, :] = out_e[M, C]
            for m_out in nl.affine_range(M):
                nisa.dma_copy(
                    dst=output[row_start + e, m_out, 0:C], src=out_e[m_out, 0:C]
                )

    return output
