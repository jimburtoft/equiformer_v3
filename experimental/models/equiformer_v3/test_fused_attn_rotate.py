"""
Test script for the fused merge+rotate NKI kernel (V3 - stacked edges).

V3 requires E divisible by EDGES_PER_TILE (128//M = 128//25 = 5).
"""

import torch
import time


def reference_merge_rotate(x_source, x_target, ew_src, ew_tgt, wigner):
    """PyTorch reference implementation."""
    # Merge
    msg = x_source * ew_src + x_target * ew_tgt
    # Wigner rotation: bmm
    output = torch.bmm(wigner, msg)
    return output


def test_kernel_device(kernel_fn, E, M, C, name="kernel", dtype=torch.float32):
    """Test a kernel with tensors on neuron device."""
    device = torch.device("privateuseone:0")
    print(f"\n{'=' * 60}")
    print(f"Testing {name}: E={E}, M={M}, C={C}, dtype={dtype}")
    print(f"{'=' * 60}")

    # Create test data on CPU first for reference
    torch.manual_seed(42)
    x_source_cpu = torch.randn(E, M, C, dtype=dtype)
    x_target_cpu = torch.randn(E, M, C, dtype=dtype)
    ew_src_cpu = torch.randn(E, M, C, dtype=dtype)
    ew_tgt_cpu = torch.randn(E, M, C, dtype=dtype)
    wigner_cpu = torch.randn(E, M, M, dtype=dtype)

    # Reference on CPU
    ref_output = reference_merge_rotate(
        x_source_cpu, x_target_cpu, ew_src_cpu, ew_tgt_cpu, wigner_cpu
    )
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Reference output range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")

    # Move to device
    x_source = x_source_cpu.to(device)
    x_target = x_target_cpu.to(device)
    ew_src = ew_src_cpu.to(device)
    ew_tgt = ew_tgt_cpu.to(device)
    wigner_d = wigner_cpu.to(device)

    # Run NKI kernel
    print(f"Compiling and running NKI kernel...")
    try:
        nki_output = kernel_fn(x_source, x_target, ew_src, ew_tgt, wigner_d)
        nki_output_cpu = nki_output.cpu()
        print(f"NKI output shape: {nki_output_cpu.shape}")
        print(
            f"NKI output range: [{nki_output_cpu.min():.4f}, {nki_output_cpu.max():.4f}]"
        )

        # Compare
        diff = (ref_output - nki_output_cpu).abs()
        print(f"Max abs diff: {diff.max():.8f}")
        print(f"Mean abs diff: {diff.mean():.8f}")

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_output.flatten().unsqueeze(0), nki_output_cpu.flatten().unsqueeze(0)
        ).item()
        print(f"Cosine similarity: {cos_sim:.10f}")

        passed = cos_sim > 0.999
        print(
            f"{'PASSED' if passed else 'FAILED'}: cos_sim {'>' if passed else '<'} 0.999"
        )
        return passed
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def benchmark_kernel(
    kernel_fn, E, M, C, name="kernel", dtype=torch.float32, warmup=3, iters=10
):
    """Benchmark kernel execution time on neuron device."""
    import torch_neuronx

    device = torch.device("privateuseone:0")
    print(f"\nBenchmarking {name}: E={E}, M={M}, C={C}")

    torch.manual_seed(42)
    x_source = torch.randn(E, M, C, dtype=dtype, device=device)
    x_target = torch.randn(E, M, C, dtype=dtype, device=device)
    ew_src = torch.randn(E, M, C, dtype=dtype, device=device)
    ew_tgt = torch.randn(E, M, C, dtype=dtype, device=device)
    wigner = torch.randn(E, M, M, dtype=dtype, device=device)

    # Warmup (includes first compilation)
    print(f"  Warming up ({warmup} iters)...")
    for _ in range(warmup):
        _ = kernel_fn(x_source, x_target, ew_src, ew_tgt, wigner)
        torch_neuronx.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        torch_neuronx.synchronize()
        t0 = time.time()
        _ = kernel_fn(x_source, x_target, ew_src, ew_tgt, wigner)
        torch_neuronx.synchronize()
        t1 = time.time()
        times.append((t1 - t0) * 1000)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    print(f"  Avg: {avg:.2f} ms, Min: {mn:.2f} ms, Max: {mx:.2f} ms")
    return avg


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/code/experimental/models/equiformer_v3")

    import torch
    import torch_neuronx

    device = torch.device("privateuseone:0")

    # Test parameters
    M = 25  # (lmax+1)^2
    C = 128  # channels

    from nki_fused_attn_rotate import nki_fused_merge_rotate

    # Phase 1: Single tile correctness test (E=128)
    print("=" * 60)
    print("PHASE 1: Correctness test (E=128, single tile)")
    print("=" * 60)
    passed = test_kernel_device(
        nki_fused_merge_rotate, E=128, M=M, C=C, name="V4 (E=128)"
    )

    # Phase 2: Multi-tile (E=256)
    if passed:
        print("\n\nPHASE 2: Multi-tile correctness (E=256)")
        print("=" * 60)
        passed = test_kernel_device(
            nki_fused_merge_rotate, E=256, M=M, C=C, name="V4 (E=256)"
        )

    # Phase 3: Non-aligned (E=300, not multiple of 128)
    if passed:
        print("\n\nPHASE 3: Non-aligned correctness (E=300)")
        print("=" * 60)
        passed = test_kernel_device(
            nki_fused_merge_rotate, E=300, M=M, C=C, name="V4 (E=300)"
        )

    # Phase 4: Production scale correctness (E=9000)
    if passed:
        print("\n\nPHASE 4: Production scale correctness (E=9000)")
        print("=" * 60)
        passed = test_kernel_device(
            nki_fused_merge_rotate, E=9000, M=M, C=C, name="V4 (E=9000)"
        )

    # Phase 5: Benchmark
    if passed:
        print("\n\nPHASE 5: Benchmark at production scale")
        print("=" * 60)

        # NKI kernel
        nki_time = benchmark_kernel(
            nki_fused_merge_rotate,
            E=9000,
            M=M,
            C=C,
            name="NKI V4 fused merge+rotate (E=9000)",
            warmup=5,
            iters=20,
        )

        # PyTorch reference on device
        def torch_merge_rotate_device(x_source, x_target, ew_src, ew_tgt, wigner):
            msg = x_source * ew_src + x_target * ew_tgt
            return torch.bmm(wigner, msg)

        pt_time = benchmark_kernel(
            torch_merge_rotate_device,
            E=9000,
            M=M,
            C=C,
            name="PyTorch eager (bmm, E=9000)",
            warmup=5,
            iters=20,
        )

        print(f"\n{'=' * 60}")
        print(f"SUMMARY: NKI V4 = {nki_time:.2f} ms, PyTorch = {pt_time:.2f} ms")
        speedup = pt_time / nki_time if nki_time > 0 else 0
        print(
            f"Speedup: {speedup:.2f}x {'(NKI wins)' if speedup > 1 else '(PyTorch wins)'}"
        )
        print(f"{'=' * 60}")

    print("\n\nDone!")
