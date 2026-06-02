"""Quick per-op benchmark without scatter."""

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


print("=== Per-Op Timing ===")
print(f"    E={E}, M={M}, C={C}, N={N}")
print()

x_node = torch.randn(N, M, C, device=device)
edge_idx = torch.randint(0, N, (E,), device=device)
x_s = torch.randn(E, M, C, device=device)
x_t = torch.randn(E, M, C, device=device)
ew_src = torch.randn(E, M, C, device=device)
ew_tgt = torch.randn(E, M, C, device=device)
wigner = torch.randn(E, M, M, device=device)
msg = torch.randn(E, M, C, device=device)
rotated = torch.randn(E, M, C, device=device)

total = 0.0
total += bench(lambda: x_node[edge_idx], "1. gather")
total += bench(lambda: x_s * ew_src + x_t * ew_tgt, "2. merge")
total += bench(lambda: torch.bmm(wigner, msg), "3. bmm_fwd")

# SO2_1 (5 matmuls)
split_sizes = [5, 8, 6, 4, 2]
w_all = [torch.randn(C, 2 * C, device=device) for _ in range(5)]


def so2_1():
    splits = rotated.split(split_sizes, dim=1)
    return torch.cat([torch.matmul(s, w_all[i]) for i, s in enumerate(splits)], dim=1)


total += bench(so2_1, "4. SO2_1 (5 matmuls)")

# SwiGLU
so2_out = torch.randn(E, M, 2 * C, device=device)


def swiglu():
    g, v = so2_out.chunk(2, dim=-1)
    return g * torch.nn.functional.silu(v)


total += bench(swiglu, "5. SwiGLU")

# SO2_2
w2_all = [torch.randn(C, C, device=device) for _ in range(5)]
act = torch.randn(E, M, C, device=device)


def so2_2():
    splits = act.split(split_sizes, dim=1)
    return torch.cat([torch.matmul(s, w2_all[i]) for i, s in enumerate(splits)], dim=1)


total += bench(so2_2, "6. SO2_2 (5 matmuls)")

# bmm inv
total += bench(lambda: torch.bmm(wigner, msg), "7. bmm_inv")

# Attention score
q = torch.randn(E, M, C, device=device)
k = torch.randn(E, M, C, device=device)


def attn_score():
    scores = (q * k).sum(dim=(1, 2))
    return torch.softmax(scores.view(N, -1), dim=-1).view(-1, 1, 1)


total += bench(attn_score, "8. attn_scores")

# attn weight * value
attn_w = torch.randn(E, 1, 1, device=device)
value = torch.randn(E, M, C, device=device)
total += bench(lambda: attn_w * value, "9. weighted_value")

print(f"\n{'=' * 50}")
print(f"Sum of isolated ops: {total:.2f} ms")
print(f"Actual layer = ~60 ms")
print(f"{'=' * 50}")

# Full pipeline
print("\n=== Full pipeline (single function, no scatter) ===")


def full_pipeline():
    xs = x_node[edge_idx]
    xt = x_node[edge_idx]
    m = xs * ew_src + xt * ew_tgt
    rot = torch.bmm(wigner, m)
    splits = rot.split(split_sizes, dim=1)
    so2_1_out = torch.cat(
        [torch.matmul(s, w_all[i]) for i, s in enumerate(splits)], dim=1
    )
    g, v = so2_1_out.chunk(2, dim=-1)
    activated = g * torch.nn.functional.silu(v)
    splits2 = activated.split(split_sizes, dim=1)
    so2_2_out = torch.cat(
        [torch.matmul(s, w2_all[i]) for i, s in enumerate(splits2)], dim=1
    )
    rotated_back = torch.bmm(wigner, so2_2_out)
    scores = (rot * rotated_back).sum(dim=(1, 2))
    aw = torch.softmax(scores.view(N, -1), dim=-1).view(-1, 1, 1)
    return aw * rotated_back


full_time = bench(full_pipeline, "full_pipeline")
print(f"\nFull pipeline time: {full_time:.2f} ms")
print(f"vs sum of isolated ops: {total:.2f} ms")
print(f"Additional overhead from sequential execution: {full_time - total:.2f} ms")
