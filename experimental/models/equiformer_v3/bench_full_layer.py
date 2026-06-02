"""
Benchmark: full forward_dense layer simulation.
Includes radial function (MLP), dense reduce, attention heads.
Goal: reproduce the ~60ms layer time and identify the bottleneck.
"""

import torch
import torch_neuronx
import time

device = torch.device("privateuseone:0")
E = 9000  # N*K = 300*30
M = 25  # (lmax+1)^2
C = 128  # channels
N = 300
K = 30
H = 4  # num attention heads
C_val = C // H  # = 32


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


print("=== Full forward_dense Simulation ===")
print(f"    N={N}, K={K}, E={E}, M={M}, C={C}, H={H}")
print()

# Pre-create tensors
x = torch.randn(N, M, C, device=device)
neighbor_idx = torch.randint(0, N, (N, K), device=device)
edge_distance = torch.randn(E, 64, device=device)  # RBF expanded
mask = torch.ones(N, K, dtype=torch.bool, device=device)
wigner = torch.randn(E, M, M, device=device)

# Radial function weights (simple 2-layer MLP: 64 -> 256 -> M*2C)
# In practice: edge_distance[E, D] -> rad_func -> [E, lmax+1, 2*C]
# Then expand_index -> [E, M, 2*C]
rad_w1 = torch.randn(64, 256, device=device)
rad_b1 = torch.randn(256, device=device)
rad_w2 = torch.randn(256, 5 * 2 * C, device=device)  # Output: [E, 5*2C] (lmax+1=5)
expand_index = torch.tensor(
    [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4],
    device=device,
)

# SO2 weights
split_sizes = [5, 8, 6, 4, 2]
w_so2_1 = [torch.randn(C, 2 * C, device=device) for _ in range(5)]
w_so2_2 = [torch.randn(C, C, device=device) for _ in range(5)]

# Attention projection
alpha_dot = torch.randn(H, C_val, device=device)

total = 0.0


# 1. Radial function (MLP)
def rad_func():
    h = torch.matmul(edge_distance, rad_w1) + rad_b1  # [E, 256]
    h = torch.nn.functional.silu(h)
    out = torch.matmul(h, rad_w2)  # [E, M*2C]
    out = out.view(E, 5, 2 * C)  # [E, lmax+1, 2*C] (5 = lmax+1)
    # expand_index: [E, 5, 2C] -> [E, 25, 2C]
    return out[:, expand_index, :]


total += bench(rad_func, "1. rad_func (MLP)")


# 2. Dense gather
def gather():
    x_source = x[neighbor_idx.view(-1)]  # [E, M, C]
    x_target = x.unsqueeze(1).expand(-1, K, -1, -1).reshape(E, M, C)
    return x_source, x_target


total += bench(gather, "2. gather (source+target)")

# 3. Merge
x_s = torch.randn(E, M, C, device=device)
x_t = torch.randn(E, M, C, device=device)
ew_src = torch.randn(E, M, C, device=device)
ew_tgt = torch.randn(E, M, C, device=device)


def merge():
    return x_s * ew_src + x_t * ew_tgt


total += bench(merge, "3. merge")

# 4. Wigner rotate
msg = torch.randn(E, M, C, device=device)
total += bench(lambda: torch.bmm(wigner, msg), "4. wigner_rot")

# 5. SO2_1 (5 matmuls, C -> 2C)
rotated = torch.randn(E, M, C, device=device)


def so2_1():
    splits = rotated.split(split_sizes, dim=1)
    return torch.cat([torch.matmul(s, w_so2_1[i]) for i, s in enumerate(splits)], dim=1)


total += bench(so2_1, "5. SO2_1 (5 matmuls)")

# 6. SwiGLU
so2_out = torch.randn(E, M, 2 * C, device=device)


def swiglu():
    g, v = so2_out.chunk(2, dim=-1)
    return g * torch.nn.functional.silu(v)


total += bench(swiglu, "6. SwiGLU")

# 7. SO2_2 (5 matmuls, C -> C)
act_out = torch.randn(E, M, C, device=device)


def so2_2():
    splits = act_out.split(split_sizes, dim=1)
    return torch.cat([torch.matmul(s, w_so2_2[i]) for i, s in enumerate(splits)], dim=1)


total += bench(so2_2, "7. SO2_2 (5 matmuls)")

# 8. Attention (alpha computation + softmax)
x_alpha = torch.randn(E, H, C_val, device=device)


def attention():
    # LayerNorm + activation on alpha (simplified)
    a = torch.nn.functional.layer_norm(x_alpha, [C_val])
    a = torch.nn.functional.silu(a)
    # Dot with alpha_dot: [E, H, A] * [H, A] -> [E, H] via einsum
    scores = torch.einsum("bik, ik -> bi", a, alpha_dot)
    # Reshape to [N, K, H], softmax over K
    scores = scores.view(N, K, H)
    scores = torch.nn.functional.softmax(scores, dim=1)
    return scores.view(E, 1, H, 1)


total += bench(attention, "8. attention (layernorm+softmax)")

# 9. Attention weighting
attn_w = torch.randn(E, 1, H, 1, device=device)
value = torch.randn(E, M, H, C_val, device=device)


def attn_weight():
    return (value * attn_w).view(E, M, C)


total += bench(attn_weight, "9. attn_weight*value")

# 10. Wigner rotate inverse
val_out = torch.randn(E, M, C, device=device)
total += bench(lambda: torch.bmm(wigner, val_out), "10. wigner_rot_inv")

# 11. Dense reduce (reshape + mask + sum)
x_msg = torch.randn(E, M, C, device=device)
mask_exp = mask.unsqueeze(-1).unsqueeze(-1)  # [N, K, 1, 1]


def dense_reduce():
    r = x_msg.view(N, K, M, C)
    r = r * mask_exp
    return r.sum(dim=1)


total += bench(dense_reduce, "11. dense_reduce (reshape+sum)")

# 12. Output projection (linear)
proj_w = torch.randn(C, C, device=device)
node_out = torch.randn(N, M, C, device=device)
total += bench(lambda: torch.matmul(node_out, proj_w), "12. output_proj")

print(f"\n{'=' * 50}")
print(f"TOTAL (sum of ops): {total:.2f} ms")
print(f"Previous actual layer time: ~60 ms")
print(f"{'=' * 50}")

# Now run all together as one function
print("\n=== Full layer as single call ===")


def full_layer():
    # Rad func
    h = torch.matmul(edge_distance, rad_w1) + rad_b1
    h = torch.nn.functional.silu(h)
    ew = torch.matmul(h, rad_w2).view(E, 5, 2 * C)
    ew = ew[:, expand_index, :]  # [E, M, 2C]
    ew_s, ew_t = ew.chunk(2, dim=-1)  # each [E, M, C]

    # Gather
    xs = x[neighbor_idx.view(-1)]
    xt = x.unsqueeze(1).expand(-1, K, -1, -1).reshape(E, M, C)

    # Merge
    m = xs * ew_s + xt * ew_t

    # Wigner
    rot = torch.bmm(wigner, m)

    # SO2_1
    splits = rot.split(split_sizes, dim=1)
    so2_1_out = torch.cat(
        [torch.matmul(s, w_so2_1[i]) for i, s in enumerate(splits)], dim=1
    )

    # SwiGLU
    g, v = so2_1_out.chunk(2, dim=-1)
    activated = g * torch.nn.functional.silu(v)

    # SO2_2
    splits2 = activated.split(split_sizes, dim=1)
    so2_2_out = torch.cat(
        [torch.matmul(s, w_so2_2[i]) for i, s in enumerate(splits2)], dim=1
    )

    # Wigner inv
    rotated_back = torch.bmm(wigner, so2_2_out)

    # Dense reduce
    r = rotated_back.view(N, K, M, C)
    r = r * mask_exp
    node_result = r.sum(dim=1)

    # Output proj
    return torch.matmul(node_result, proj_w)


full_time = bench(full_layer, "full_layer")
print(f"\nFull layer: {full_time:.2f} ms")
