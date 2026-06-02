"""
Benchmark: selective torch.compile vs eager vs fused_linear on EquiformerV3.

Tests the EquivariantGraphAttention.forward_dense with three strategies:
1. Eager (original SO2 forward with subtract - catastrophically slow)
2. Eager + fused_linear (manual NCC_ILSA902 workaround)
3. Selective compile (element-wise + SO2 compiled, bmm/gather in eager)
"""

import sys

sys.path.insert(0, "/code")
import torch
import torch_neuronx
import time

device = torch.device("privateuseone:0")
E = 9000
M = 25
C = 128
N = 300
K = 30

from experimental.models.equiformer_v3.so2_ops import SO2Linear

# Create SO2 layers matching the real model
so2_1 = SO2Linear(num_in_channels=C, num_out_channels=C, lmax=4, mmax=4)
so2_2 = SO2Linear(num_in_channels=C, num_out_channels=C, lmax=4, mmax=4)
so2_1 = so2_1.to(device).eval()
so2_2 = so2_2.to(device).eval()

# Inputs
x = torch.randn(N, M, C, device=device)
wigner = torch.randn(E, M, M, device=device)
wigner_inv = torch.randn(E, M, M, device=device)
neighbor_idx = torch.randint(0, N, (N, K), device=device)
ew_src = torch.randn(E, M, C, device=device)
ew_tgt = torch.randn(E, M, C, device=device)
mask = torch.ones(N, K, dtype=torch.bool, device=device)


def bench(fn, name, warmup=5, iters=20):
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
            torch_neuronx.synchronize()
        times = []
        for _ in range(iters):
            torch_neuronx.synchronize()
            t0 = time.time()
            _ = fn()
            torch_neuronx.synchronize()
            times.append((time.time() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    print(f"  {name}: avg={avg:.2f} ms, min={mn:.2f} ms")
    return avg


# ===== Strategy 1: Fully Eager (original SO2 with subtract) =====
def full_eager():
    x_source = x[neighbor_idx.view(-1)]
    x_target = x.unsqueeze(1).expand(-1, K, -1, -1).reshape(E, M, C)
    x_msg = x_source * ew_src + x_target * ew_tgt
    x_msg = torch.bmm(wigner, x_msg)
    x_msg = so2_1(x_msg)
    x_msg = x_msg * torch.sigmoid(x_msg)  # SiLU
    x_msg = so2_2(x_msg)
    x_msg = torch.bmm(wigner_inv, x_msg)
    x_msg = x_msg.view(N, K, M, C)
    x_msg = x_msg * mask.unsqueeze(-1).unsqueeze(-1)
    return x_msg.sum(dim=1)


print("=" * 60)
print(f"EquiformerV3 Attention Layer Benchmark")
print(f"  N={N}, K={K}, E={E}, M={M}, C={C}")
print("=" * 60)

print("\n[1/3] Eager (original SO2, with subtract issue)...")
t_eager = bench(full_eager, "EAGER (original)")


# ===== Strategy 2: Eager + Fused Linear =====
so2_1.build_fused_linear()
so2_2.build_fused_linear()


def full_fused():
    x_source = x[neighbor_idx.view(-1)]
    x_target = x.unsqueeze(1).expand(-1, K, -1, -1).reshape(E, M, C)
    x_msg = x_source * ew_src + x_target * ew_tgt
    x_msg = torch.bmm(wigner, x_msg)
    x_msg = so2_1.forward_fused(x_msg)
    x_msg = x_msg * torch.sigmoid(x_msg)
    x_msg = so2_2.forward_fused(x_msg)
    x_msg = torch.bmm(wigner_inv, x_msg)
    x_msg = x_msg.view(N, K, M, C)
    x_msg = x_msg * mask.unsqueeze(-1).unsqueeze(-1)
    return x_msg.sum(dim=1)


print("\n[2/3] Eager + fused_linear (NCC_ILSA902 workaround)...")
t_fused = bench(full_fused, "EAGER + fused SO2")


# ===== Strategy 3: Hybrid (fused SO2 + compiled element-wise) =====
print("\n[3/3] Hybrid: fused SO2 + compiled element-wise...")
print("  Compiling merge...")
merge_compiled = torch.compile(
    lambda xs, xt, es, et: xs * es + xt * et, backend="neuron"
)
print("  Compiling SiLU...")
silu_compiled = torch.compile(lambda x_in: x_in * torch.sigmoid(x_in), backend="neuron")

# Warmup compiled kernels
print("  Warming up compiled kernels...")
with torch.no_grad():
    dummy = torch.randn(E, M, C, device=device)
    _ = merge_compiled(dummy, dummy, dummy, dummy)
    _ = silu_compiled(dummy)
    torch_neuronx.synchronize()
    for _ in range(3):
        _ = merge_compiled(dummy, dummy, dummy, dummy)
        _ = silu_compiled(dummy)
        torch_neuronx.synchronize()


def full_hybrid():
    x_source = x[neighbor_idx.view(-1)]
    x_target = x.unsqueeze(1).expand(-1, K, -1, -1).reshape(E, M, C)
    # Compiled merge (2.4x speedup)
    x_msg = merge_compiled(x_source, x_target, ew_src, ew_tgt)
    # Eager bmm (fastest for batched matmul)
    x_msg = torch.bmm(wigner, x_msg)
    # Fused SO2 (faster than compiled for matmuls)
    x_msg = so2_1.forward_fused(x_msg)
    # Compiled SiLU (faster than eager)
    x_msg = silu_compiled(x_msg)
    # Fused SO2
    x_msg = so2_2.forward_fused(x_msg)
    # Eager bmm
    x_msg = torch.bmm(wigner_inv, x_msg)
    # Eager aggregate
    x_msg = x_msg.view(N, K, M, C)
    x_msg = x_msg * mask.unsqueeze(-1).unsqueeze(-1)
    return x_msg.sum(dim=1)


t_compiled = bench(full_hybrid, "HYBRID (fused SO2 + compiled elem)")


# ===== Summary =====
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Eager (original SO2):     {t_eager:.2f} ms")
print(
    f"  Eager + fused SO2:        {t_fused:.2f} ms  ({t_eager / t_fused:.1f}x vs baseline)"
)
print(
    f"  Hybrid (best of both):    {t_compiled:.2f} ms  ({t_eager / t_compiled:.1f}x vs baseline, {t_fused / t_compiled:.2f}x vs fused)"
)
print()
print(f"  Hybrid saves: {t_fused - t_compiled:.2f} ms per layer vs fused-only")
print(f"  Over 8 layers: {(t_fused - t_compiled) * 8:.1f} ms total savings")
print("=" * 60)
