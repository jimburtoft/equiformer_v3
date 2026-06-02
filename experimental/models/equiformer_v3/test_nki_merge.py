"""Test and benchmark the merge-only NKI kernel vs PyTorch eager and torch.compile."""

import torch
import torch_neuronx
import time
import sys

sys.path.insert(0, "/code/experimental/models/equiformer_v3")

device = torch.device("privateuseone:0")
E, M, C = 9000, 25, 128


def bench(fn, name, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
        torch_neuronx.synchronize()
    times = []
    for _ in range(iters):
        torch_neuronx.synchronize()
        t0 = time.time()
        fn()
        torch_neuronx.synchronize()
        times.append((time.time() - t0) * 1000)
    avg = sum(times) / len(times)
    print(f"  {name}: avg={avg:.3f} ms, min={min(times):.3f} ms")
    return avg


# Create tensors
torch.manual_seed(42)
x_source = torch.randn(E, M, C, device=device)
x_target = torch.randn(E, M, C, device=device)
ew_src = torch.randn(E, M, C, device=device)
ew_tgt = torch.randn(E, M, C, device=device)

print(f"=== Merge benchmark: E={E}, M={M}, C={C} ===")
print(f"    Tensor shape: [{E}, {M}, {C}]")
print(f"    Total elements: {E * M * C:,} ({E * M * C * 4 / 1e6:.1f} MB per tensor)")
print()

# 1. PyTorch eager
print("--- PyTorch eager ---")
eager_time = bench(lambda: x_source * ew_src + x_target * ew_tgt, "eager merge")

# 2. torch.compile(backend='neuron')
print("\n--- torch.compile(backend='neuron') ---")


@torch.compile(backend="neuron")
def compiled_merge(xs, xt, ews, ewt):
    return xs * ews + xt * ewt


# Trigger compilation
_ = compiled_merge(x_source, x_target, ew_src, ew_tgt)
torch_neuronx.synchronize()
compile_time = bench(
    lambda: compiled_merge(x_source, x_target, ew_src, ew_tgt), "compiled merge"
)

# 3. NKI kernel
print("\n--- NKI kernel (P=128, full partition) ---")
from nki_fused_merge import nki_fused_merge

# First call triggers compilation
print("  Compiling...")
t0 = time.time()
nki_out = nki_fused_merge(x_source, x_target, ew_src, ew_tgt)
torch_neuronx.synchronize()
print(f"  Compile time: {(time.time() - t0) * 1000:.0f} ms")

# Correctness check
ref = (x_source * ew_src + x_target * ew_tgt).cpu()
nki_cpu = nki_out.cpu()
cos_sim = torch.nn.functional.cosine_similarity(
    ref.flatten().unsqueeze(0), nki_cpu.flatten().unsqueeze(0)
).item()
max_diff = (ref - nki_cpu).abs().max().item()
print(f"  Correctness: cos_sim={cos_sim:.10f}, max_diff={max_diff:.8f}")

if cos_sim < 0.999:
    print("  FAILED correctness check!")
else:
    nki_time = bench(
        lambda: nki_fused_merge(x_source, x_target, ew_src, ew_tgt), "NKI merge"
    )

    print(f"\n{'=' * 50}")
    print(f"SUMMARY:")
    print(f"  PyTorch eager:    {eager_time:.3f} ms")
    print(f"  torch.compile:    {compile_time:.3f} ms")
    print(f"  NKI kernel:       {nki_time:.3f} ms")
    print(f"  NKI vs eager:     {eager_time / nki_time:.2f}x")
    print(f"  NKI vs compile:   {compile_time / nki_time:.2f}x")
    print(f"{'=' * 50}")
