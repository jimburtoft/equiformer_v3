"""Benchmark: Full 8-layer EquiformerV3 inference with hybrid optimization.

Uses the dense padded forward path (forward_dense) which avoids scatter_reduce
and uses F.softmax over the fixed neighbor dimension — compatible with Neuron.

Tests three modes at production scale (N=300, K=30, E=9000):
1. Eager (original SO2 with subtract - catastrophically slow)
2. Eager + fused SO2 (build_fused_linear)
3. Hybrid (enable_compile: fused SO2 + compiled element-wise)

Usage (in Docker container on trn2):
    python bench_full_model_inference.py
"""

import sys, os, types, contextlib, time

sys.path.insert(0, "/code")

import torch
import torch_neuronx

# Fairchem stubs
for pkg_name in [
    "fairchem",
    "fairchem.core",
    "fairchem.core.common",
    "fairchem.core.models",
]:
    mod = types.ModuleType(pkg_name)
    mod.__package__ = pkg_name
    mod.__path__ = []
    sys.modules[pkg_name] = mod
    parts = pkg_name.split(".")
    if len(parts) > 1:
        parent_name = ".".join(parts[:-1])
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], parts[-1], mod)

registry_mod = types.ModuleType("fairchem.core.common.registry")
registry_mod.__package__ = "fairchem.core.common"


class Registry:
    @staticmethod
    def register_model(name):
        return lambda cls: cls


registry_mod.registry = Registry()
sys.modules["fairchem.core.common.registry"] = registry_mod
sys.modules["fairchem.core.common"].registry = registry_mod

utils_mod = types.ModuleType("fairchem.core.common.utils")
utils_mod.__package__ = "fairchem.core.common"


@contextlib.contextmanager
def conditional_grad(flag):
    if flag:
        yield
    else:
        with torch.no_grad():
            yield


utils_mod.conditional_grad = conditional_grad
sys.modules["fairchem.core.common.utils"] = utils_mod
sys.modules["fairchem.core.common"].utils = utils_mod

base_mod = types.ModuleType("fairchem.core.models.base")
base_mod.__package__ = "fairchem.core.models"


class GraphModelMixin:
    def generate_graph(self, data, **kwargs):
        raise NotImplementedError


base_mod.GraphModelMixin = GraphModelMixin
sys.modules["fairchem.core.models.base"] = base_mod
sys.modules["fairchem.core.models"].base = base_mod

sys.path.insert(0, "/code/experimental/models")

# Fix scatter_ops
import equiformer_v3.scatter_ops as scatter_ops_mod


def scatter_add_plain(src, index, output_size):
    output = torch.zeros(
        output_size, *src.shape[1:], device=src.device, dtype=src.dtype
    )
    output.index_add_(0, index, src)
    return output


scatter_ops_mod.scatter_add = scatter_add_plain

from equiformer_v3.equiformer_v3 import EquiformerV3_OC
from equiformer_v3.graph_padding import create_random_graph
from equiformer_v3.edge_rot_mat import init_edge_rot_mat

device = torch.device("privateuseone:0")

# ============================================================
# CONFIG
# ============================================================
NUM_LAYERS = 8
LMAX = 4
MMAX = 4
C = 128
N_ATOMS = 300
K_NEIGHBORS = 30
WARMUP = 3
ITERS = 10

print("=" * 70)
print("BENCHMARK: Full EquiformerV3 Inference (Hybrid Optimization)")
print(f"  {NUM_LAYERS} layers, lmax={LMAX}, mmax={MMAX}, C={C}")
print(f"  N={N_ATOMS} atoms, K={K_NEIGHBORS} neighbors, E={N_ATOMS * K_NEIGHBORS}")
print("=" * 70)

# ============================================================
# CREATE MODEL
# ============================================================
print("\nCreating model...")
t0 = time.time()
model = EquiformerV3_OC(
    direct_prediction=True,
    regress_forces=True,
    regress_stress=False,
    num_layers=NUM_LAYERS,
    num_channels=C,
    lmax=LMAX,
    mmax=MMAX,
    num_radial_basis=32,
    max_neighbors=K_NEIGHBORS,
    max_radius=12.0,
    num_heads=8,
    attn_hidden_channels=64,
    attn_alpha_channels=32,
    attn_value_channels=16,
    ffn_hidden_channels=128,
    use_grid_mlp=True,
    use_envelope=True,
    alpha_drop=0.0,
    attn_mask_rate=0.0,
    attn_weights_drop=0.0,
    value_drop=0.0,
    drop_path_rate=0.0,
    proj_drop=0.0,
    ffn_drop=0.0,
    use_pbc=False,
    otf_graph=False,
    use_atom_edge_embedding=True,
    use_attn_renorm=True,
    use_add_merge=True,  # Additive merge (saves 2x on wigner rotation)
    use_rad_l_parametrization=True,
    norm_type="merge_layer_norm",
    attn_activation="sep-merge_gates2_swiglu",
    ffn_activation="sep-merge_gates2_swiglu",
    use_gate_force_head=True,
).to(device=device, dtype=torch.float32)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"  Model created in {time.time() - t0:.1f}s")
print(f"  Parameters: {num_params:,} ({num_params * 4 / 1e6:.1f} MB FP32)")

# ============================================================
# CREATE TEST DATA (dense padded format for forward_dense)
# ============================================================
print(f"\nCreating test data (N={N_ATOMS}, K={K_NEIGHBORS})...")

# Create random graph on CPU first (wigner computation needs atan2/acos on CPU)
graph = create_random_graph(
    num_atoms=N_ATOMS,
    max_neighbors=K_NEIGHBORS,
    max_radius=12.0,
    batch_size=1,
    device="cpu",
    dtype=torch.float32,
)

# Build dense neighbor_idx [N, K] from edge_index
N = N_ATOMS
K = K_NEIGHBORS
edge_index = graph["edge_index"]
positions = graph["positions"]

# For each atom, find its neighbors (pad with 0 if fewer than K)
neighbor_idx = torch.zeros(N, K, dtype=torch.long)
neighbor_count = torch.zeros(N, dtype=torch.long)
for e in range(edge_index.shape[1]):
    dst = edge_index[1, e].item()  # target node
    src = edge_index[0, e].item()  # source node (neighbor)
    idx = neighbor_count[dst].item()
    if idx < K:
        neighbor_idx[dst, idx] = src
        neighbor_count[dst] += 1

# Create mask: True where we have a real neighbor
mask = torch.zeros(N, K, dtype=torch.bool)
for i in range(N):
    mask[i, : neighbor_count[i].item()] = True

# Compute edge distance vectors in dense layout [N*K, 3]
src_positions = positions[neighbor_idx.view(-1)]  # [N*K, 3]
dst_positions = positions.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)  # [N*K, 3]
edge_distance_vec_dense = src_positions - dst_positions  # [N*K, 3]
edge_distance_dense = edge_distance_vec_dense.norm(dim=-1)  # [N*K]

# Zero out padding edges
flat_mask = mask.view(-1)
edge_distance_vec_dense[~flat_mask] = 0.0
edge_distance_dense[~flat_mask] = 0.0

# Compute wigner matrices on CPU (requires atan2/acos)
print("  Computing Wigner-D matrices on CPU...")
edge_rot_mat = init_edge_rot_mat(edge_distance_vec_dense, use_rotation_mask=False)
# Use the model's SO3Rotation to compute wigner from rot_mat
model_cpu = model.to("cpu")
model_cpu.so3_rotation.set_wigner(edge_rot_mat)
wigner = model_cpu.so3_rotation.wigner.clone()  # [N*K, M, M]
wigner_inv = model_cpu.so3_rotation.wigner_inv.clone()  # [N*K, M, M]
print(f"  Wigner shape: {wigner.shape}")

# Pre-expand weights ON CPU (eliminates index_select in SO3Linear)
print("  Pre-expanding weights (on CPU)...")
model_cpu.pre_expand_weights()

# Build fused linear ON CPU (eliminates subtract in SO2MLinear)
print("  Building fused linear layers (on CPU)...")
for i, block in enumerate(model_cpu.blocks):
    block.build_fused_linear()
print(f"  All {NUM_LAYERS} blocks: fused")

# Move model to device
print("  Moving model to device...")
model = model_cpu.to(device)

# Prepare dense inputs on device
atomic_numbers = graph["atomic_numbers"].to(device)
neighbor_idx_dev = neighbor_idx.to(device)
edge_distance_dev = edge_distance_dense.to(device)
edge_distance_vec_dev = edge_distance_vec_dense.to(device)
wigner_dev = wigner.to(device)
wigner_inv_dev = wigner_inv.to(device)
mask_dev = mask.to(device)
batch_dev = graph["batch"].to(device)

# Expand radial basis for edge_distance (needed by forward_dense internally)
# forward_dense takes raw scalar distances and computes radial basis inside
print("  Data ready on device.")


def run_inference():
    return model.forward_dense(
        atomic_numbers,
        neighbor_idx_dev,
        edge_distance_dev,
        edge_distance_vec_dev,
        wigner_dev,
        wigner_inv_dev,
        mask_dev,
        batch_dev,
        batch_size=1,
    )


def bench(name, warmup=WARMUP, iters=ITERS):
    with torch.no_grad():
        for _ in range(warmup):
            _ = run_inference()
            torch_neuronx.synchronize()
        times = []
        for _ in range(iters):
            torch_neuronx.synchronize()
            t0 = time.time()
            _ = run_inference()
            torch_neuronx.synchronize()
            times.append((time.time() - t0) * 1000)
    avg = sum(times) / len(times)
    mn = min(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {name}: avg={avg:.1f} ms, min={mn:.1f} ms, std={std:.1f} ms")
    return avg


# ============================================================
# BENCHMARK 1: Eager + fused SO2 (already built above)
# ============================================================
print("\n[1/2] Eager + fused SO2 (build_fused_linear already applied)...")
t_fused = bench("EAGER + fused SO2")

# ============================================================
# BENCHMARK 2: Hybrid (fused SO2 + compiled element-wise)
# ============================================================
print("\n[2/2] Applying enable_compile() to all layers...")
for i, block in enumerate(model.blocks):
    block.enable_compile()
    print(f"  Block {i}: compiled")

print("  Warming up compiled kernels (first pass triggers compilation)...")
with torch.no_grad():
    _ = run_inference()
    torch_neuronx.synchronize()
print("  Compilation complete.")

t_hybrid = bench("HYBRID (fused SO2 + compiled elem)")

# ============================================================
# SUMMARY
# ============================================================
# Per-layer benchmark measured original eager at 746ms/layer => ~5970ms for 8 layers
t_eager_estimated = 746.0 * NUM_LAYERS  # From per-layer bench_compile.py

print("\n" + "=" * 70)
print("SUMMARY: Full Model Inference")
print(f"  Model: {NUM_LAYERS} layers, lmax={LMAX}, C={C}, {num_params:,} params")
print(f"  Input: N={N_ATOMS}, K={K_NEIGHBORS}, E={N_ATOMS * K_NEIGHBORS}")
print("=" * 70)
print(
    f"  Eager (original SO2)*: ~{t_eager_estimated:.0f} ms  (*estimated from per-layer benchmark)"
)
print(
    f"  Eager + fused SO2:     {t_fused:.1f} ms  ({t_eager_estimated / t_fused:.1f}x vs original)"
)
print(
    f"  Hybrid (best):         {t_hybrid:.1f} ms  ({t_eager_estimated / t_hybrid:.1f}x vs original, {t_fused / t_hybrid:.2f}x vs fused)"
)
print()
print(f"  Hybrid saves: {t_fused - t_hybrid:.1f} ms vs fused-only")
print(f"  Throughput: {1000 / t_hybrid:.1f} inferences/sec")
print("=" * 70)
print(f"  Eager (original SO2):  {t_eager:.1f} ms")
print(
    f"  Eager + fused SO2:     {t_fused:.1f} ms  ({t_eager / t_fused:.1f}x vs baseline)"
)
print(
    f"  Hybrid (best):         {t_hybrid:.1f} ms  ({t_eager / t_hybrid:.1f}x vs baseline, {t_fused / t_hybrid:.2f}x vs fused)"
)
print()
print(f"  Hybrid saves: {t_fused - t_hybrid:.1f} ms vs fused-only")
print(f"  Throughput: {1000 / t_hybrid:.1f} inferences/sec")
print("=" * 70)
