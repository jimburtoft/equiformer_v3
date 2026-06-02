"""
Benchmark individual operations in the EquiformerV3 attention layer.
Purpose: understand where time goes and quantify kernel launch overhead.
"""

import torch
import torch_neuronx
import time

device = torch.device("privateuseone:0")
E, M, C = 9000, 25, 128
N = 300


def bench(fn, name, warmup=3, iters=10):
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
    print(f"  {name}: {avg:.2f} ms (min={min(times):.2f})")
    return avg


print("=== EquiformerV3 Attention Layer - Per-Op Timing ===")
print(f"    E={E}, M={M}, C={C}, N={N}")
print()

# Create tensors
x_node = torch.randn(N, M, C, device=device)
edge_idx = torch.randint(0, N, (E,), device=device)
x_s = torch.randn(E, M, C, device=device)
x_t = torch.randn(E, M, C, device=device)
ew_src = torch.randn(E, M, C, device=device)
ew_tgt = torch.randn(E, M, C, device=device)
wigner = torch.randn(E, M, M, device=device)
msg = torch.randn(E, M, C, device=device)

total = 0.0

# 1. Gather
total += bench(lambda: x_node[edge_idx], "1. gather")

# 2. Merge (element-wise)
total += bench(lambda: x_s * ew_src + x_t * ew_tgt, "2. merge")

# 3. Wigner rotate forward
total += bench(lambda: torch.bmm(wigner, msg), "3. bmm_fwd")

# 4. SO2_1 m=0 matmul: [E, 5, 128] @ [128, 256]
rotated = torch.randn(E, M, C, device=device)
m0 = rotated[:, :5, :].contiguous()
w0 = torch.randn(C, 2 * C, device=device)
total += bench(lambda: torch.matmul(m0, w0), "4. SO2_1 m=0 [E,5,C]@[C,2C]")

# 5. SO2_1 m=1 matmul: [E, 8, 128] @ [128, 256]
m1 = rotated[:, 5:13, :].contiguous()
w1 = torch.randn(C, 2 * C, device=device)
total += bench(lambda: torch.matmul(m1, w1), "5. SO2_1 m=1 [E,8,C]@[C,2C]")

# 6. SwiGLU
so2_out = torch.randn(E, M, 2 * C, device=device)


def swiglu():
    g, v = so2_out.chunk(2, dim=-1)
    return g * torch.nn.functional.silu(v)


total += bench(swiglu, "6. SwiGLU")

# 7. SO2_2 m=0 matmul: [E, 5, 128] @ [128, 128]
act = torch.randn(E, 5, C, device=device)
w2 = torch.randn(C, C, device=device)
total += bench(lambda: torch.matmul(act, w2), "7. SO2_2 m=0 [E,5,C]@[C,C]")

# 8. Wigner rotate inverse
value = torch.randn(E, M, C, device=device)
total += bench(lambda: torch.bmm(wigner, value), "8. bmm_inv")

# 9. Scatter reduce
idx_exp = edge_idx.unsqueeze(1).unsqueeze(2).expand(-1, M, C)
out_scatter = torch.randn(E, M, C, device=device)
total += bench(
    lambda: torch.zeros(N, M, C, device=device).scatter_add_(0, idx_exp, out_scatter),
    "9. scatter_add",
)


# 10. torch.compile on merge only (element-wise fusion)
@torch.compile(backend="neuron")
def compiled_merge(xs, xt, ews, ewt):
    return xs * ews + xt * ewt


# Trigger compilation
_ = compiled_merge(x_s, x_t, ew_src, ew_tgt)
torch_neuronx.synchronize()
total_c = bench(lambda: compiled_merge(x_s, x_t, ew_src, ew_tgt), "10. compiled_merge")

print(f"\n{'=' * 50}")
print(f"Sum of ops 1-9 (eager): {total:.2f} ms")
print(f"Compiled merge (op 10): {total_c:.2f} ms vs eager merge: see op 2")
print(f"Recall: actual full layer = ~60 ms")
print(
    f"Overhead estimate: {60.7 - total:.1f} ms ({(60.7 - total) / 60.7 * 100:.0f}% of layer time)"
)
print(f"{'=' * 50}")

# Now time the full pipeline as a single call
print("\n=== Full pipeline (single function) ===")

w_all = [torch.randn(C, 2 * C, device=device) for _ in range(5)]
w2_all = [torch.randn(C, C, device=device) for _ in range(5)]
split_sizes = [5, 8, 6, 4, 2]


def full_pipeline():
    # Gather
    xs = x_node[edge_idx]
    xt = x_node[edge_idx]
    # Merge
    m = xs * ew_src + xt * ew_tgt
    # Wigner
    rot = torch.bmm(wigner, m)
    # SO2_1
    splits = rot.split(split_sizes, dim=1)
    so2_1 = torch.cat([torch.matmul(s, w_all[i]) for i, s in enumerate(splits)], dim=1)
    # SwiGLU
    g, v = so2_1.chunk(2, dim=-1)
    activated = g * torch.nn.functional.silu(v)
    # SO2_2
    splits2 = activated.split(split_sizes, dim=1)
    so2_2 = torch.cat(
        [torch.matmul(s, w2_all[i]) for i, s in enumerate(splits2)], dim=1
    )
    # Wigner inv
    rotated_back = torch.bmm(wigner, so2_2)
    # Scatter
    return torch.zeros(N, M, C, device=device).scatter_add_(0, idx_exp, rotated_back)


full_time = bench(full_pipeline, "full_pipeline_eager")
print(f"\nFull pipeline: {full_time:.2f} ms")
print(f"This represents the REAL layer time with all ops in sequence")
