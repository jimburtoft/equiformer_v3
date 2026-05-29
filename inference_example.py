"""
EquiformerV3 compiled inference on AWS Trainium (Neuron).

Demonstrates the optimized inference pipeline:
1. Create model
2. build_fused_linear() - workaround for NCC_ILSA902 compiler bug
3. pre_expand_weights() - eliminate runtime index_select for compilation
4. Move to Neuron device
5. compile_for_neuron() - 15x speedup via aot_autograd + Neuron compiler
6. forward_compiled() - fast inference

Performance (2-layer, lmax=2, C=64 model):
  - Compiled: 12-17 ms (depending on input size)
  - Eager: 187 ms
  - Speedup: 11-15x

Performance (7-layer production, lmax=4, C=128):
  - Compiled: 107-515 ms (depending on edge count)
  - Eager: 770+ ms
  - Speedup: 3-4x

Usage:
    docker run --rm --device=/dev/neuron0 \
      -v ~/code:/code \
      equiformer_v3:latest \
      python /code/inference_example.py
"""

import sys
import time
import types
import contextlib
import torch

# --- Stub out fairchem (only needed for registry/base class, not inference) ---
for p in ["fairchem", "fairchem.core", "fairchem.core.common", "fairchem.core.models"]:
    m = types.ModuleType(p)
    m.__package__ = p
    m.__path__ = []
    sys.modules[p] = m
    pts = p.split(".")
    if len(pts) > 1:
        setattr(sys.modules[".".join(pts[:-1])], pts[-1], m)

r = types.ModuleType("fairchem.core.common.registry")
r.__package__ = "fairchem.core.common"


class _R:
    @staticmethod
    def register_model(n):
        return lambda c: c


r.registry = _R()
sys.modules["fairchem.core.common.registry"] = r
sys.modules["fairchem.core.common"].registry = r

u = types.ModuleType("fairchem.core.common.utils")
u.__package__ = "fairchem.core.common"


@contextlib.contextmanager
def _cg(f):
    if f:
        yield
    else:
        with torch.no_grad():
            yield


u.conditional_grad = _cg
sys.modules["fairchem.core.common.utils"] = u
sys.modules["fairchem.core.common"].utils = u

b = types.ModuleType("fairchem.core.models.base")
b.__package__ = "fairchem.core.models"


class _G:
    pass


b.GraphModelMixin = _G
sys.modules["fairchem.core.models.base"] = b
sys.modules["fairchem.core.models"].base = b

# --- Import EquiformerV3 ---
# Adjust this path if your repo is mounted differently
sys.path.insert(0, "/root/equiformer_v3")
from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
from experimental.models.equiformer_v3.edge_rot_mat import init_edge_rot_mat
from experimental.models.equiformer_v3.so3 import SO3Rotation


def create_model():
    """Create the EquiformerV3 model configured for inference."""
    model = EquiformerV3_OC(
        direct_prediction=True,
        regress_forces=False,
        regress_stress=False,
        num_layers=2,
        num_channels=64,
        lmax=2,
        mmax=2,
        num_radial_basis=32,
        max_neighbors=50,
        max_radius=12.0,
        num_heads=4,
        attn_hidden_channels=32,
        attn_alpha_channels=16,
        attn_value_channels=8,
        ffn_hidden_channels=64,
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
        use_add_merge=True,
        use_rad_l_parametrization=True,
        norm_type="merge_layer_norm",
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        use_gate_force_head=False,
    )
    model.eval()
    # CRITICAL: Build fused linear to avoid NCC_ILSA902 compiler bug.
    # Call this AFTER loading pre-trained weights.
    model.build_fused_linear()
    return model


def prepare_inputs(atomic_numbers, positions, max_neighbors, device):
    """
    Convert atomic positions to dense-padded model inputs.

    This preprocessing runs on CPU (uses atan2/acos for Wigner matrices).
    The outputs are then transferred to the Neuron device for compiled inference.

    Args:
        atomic_numbers: [N] tensor of atomic numbers (int, 1-94)
        positions: [N, 3] tensor of atom positions in Angstroms
        max_neighbors: int, K (pad neighbor list to this size)
        device: target device string (e.g., 'neuron:0')

    Returns:
        dict of inputs ready for model.forward_dense()
    """
    N = atomic_numbers.shape[0]
    K = max_neighbors

    # Simple distance-based neighbor list (replace with your own)
    # In production, use a proper neighbor list algorithm (e.g., from ASE or pymatgen)
    dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
    dists.fill_diagonal_(float("inf"))

    neighbor_idx = torch.zeros(N, K, dtype=torch.long)
    mask = torch.zeros(N, K, dtype=torch.bool)

    for i in range(N):
        # Get K nearest neighbors
        actual_k = min(K, N - 1)
        _, top_k = dists[i].topk(actual_k, largest=False)
        neighbor_idx[i, :actual_k] = top_k
        mask[i, :actual_k] = True

    # Compute edge vectors and distances
    src_pos = positions[neighbor_idx.view(-1)]
    tgt_pos = positions.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
    edge_vec = src_pos - tgt_pos
    edge_dist = edge_vec.norm(dim=1).clamp(min=0.1)

    # Zero out padding edges
    flat_mask = mask.view(-1)
    edge_dist[~flat_mask] = 1.0
    edge_vec[~flat_mask] = torch.tensor([0.0, 0.0, 1.0])

    # Pre-compute Wigner matrices (CPU — requires atan2/acos)
    lmax, mmax = 2, 2  # Must match model config
    temp_so3 = SO3Rotation(lmax, mmax, use_rotation_mask=False)
    with torch.no_grad():
        edge_rot_mat = init_edge_rot_mat(edge_vec, use_rotation_mask=False)
        temp_so3.set_wigner(edge_rot_mat)
        wigner = temp_so3.wigner.clone()
        wigner_inv = temp_so3.wigner_inv.clone()

    # Transfer to device
    inputs = dict(
        atomic_numbers=atomic_numbers.to(device),
        neighbor_idx=neighbor_idx.to(device),
        edge_distance=edge_dist.to(device),
        edge_distance_vec=edge_vec.to(device),
        wigner=wigner.to(device),
        wigner_inv=wigner_inv.to(device),
        mask=mask.to(device),
        batch=torch.zeros(N, dtype=torch.long).to(device),
        batch_size=1,
    )
    return inputs


def main():
    # --- Configuration ---
    NUM_ATOMS = 100
    MAX_NEIGHBORS = 10
    DEVICE = "neuron"

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"Config: {NUM_ATOMS} atoms, {MAX_NEIGHBORS} max neighbors")
    print()

    # --- Create Model ---
    print("Creating model...")
    model = create_model()

    # CRITICAL: Prepare model for Neuron compilation (call in this order)
    # 1. build_fused_linear: workaround for NCC_ILSA902 compiler bug
    # 2. pre_expand_weights: eliminate runtime index_select for compilability
    model.build_fused_linear()
    model.pre_expand_weights()
    model = model.to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Prepare Inputs ---
    # For forward_static, we need: atomic_numbers, edge_index, edge_distance,
    # edge_distance_vec, batch, batch_size
    print("Preparing inputs...")
    torch.manual_seed(42)

    from experimental.models.equiformer_v3.graph_padding import create_random_graph

    graph = create_random_graph(
        num_atoms=NUM_ATOMS,
        max_neighbors=MAX_NEIGHBORS,
        max_radius=12.0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    data = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in graph.items()
    }
    edges = graph["edge_index"].shape[1]
    print(f"  Total edges: {edges}")

    # --- Compile with aot_autograd (15x speedup over eager) ---
    print("\nCompiling model (first call: ~30-60s, cached afterward)...")
    t0 = time.time()
    warmup_input = (
        data["atomic_numbers"],
        data["edge_index"],
        data["edge_distance"],
        data["edge_distance_vec"],
        data["batch"],
        data["batch_size"],
    )
    compiled_fn = model.compile_for_neuron(warmup_input=warmup_input)
    compile_time = time.time() - t0
    print(f"  Done in {compile_time:.1f}s")

    # --- Warmup ---
    print("Warming up (5 iterations)...")
    with torch.no_grad():
        for _ in range(5):
            model.forward_compiled(*warmup_input)

    # --- Benchmark ---
    print("Benchmarking (20 iterations)...")
    times = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter_ns()
            output = model.forward_compiled(*warmup_input)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1e6)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    print(f"\n{'=' * 60}")
    print(f" INFERENCE RESULTS (compiled)")
    print(f"{'=' * 60}")
    print(f"  Atoms:       {NUM_ATOMS}")
    print(f"  Neighbors:   {MAX_NEIGHBORS}")
    print(f"  Total edges: {edges}")
    print(f"  Avg latency: {avg_ms:.1f} ms")
    print(f"  Min latency: {min_ms:.1f} ms")
    print(f"  Max latency: {max_ms:.1f} ms")
    print(f"  Throughput:  {edges / avg_ms:.1f} edges/ms")
    print(f"  Throughput:  {1000 / avg_ms:.1f} inferences/s")
    print(f"  Output:      energy = {output['energy'].cpu().item():.6f}")
    print(f"               forces shape = {output['forces'].shape}")
    print(f"  Compile time: {compile_time:.1f}s (one-time cost)")
    print(f"{'=' * 60}")

    # --- CPU Comparison ---
    print("\nRunning CPU comparison...")
    model_cpu = create_model()
    model_cpu.build_fused_linear()
    model_cpu.pre_expand_weights()

    graph_cpu = create_random_graph(
        num_atoms=NUM_ATOMS,
        max_neighbors=MAX_NEIGHBORS,
        max_radius=12.0,
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )

    with torch.no_grad():
        for _ in range(3):
            model_cpu.forward_static(
                graph_cpu["atomic_numbers"],
                graph_cpu["edge_index"],
                graph_cpu["edge_distance"],
                graph_cpu["edge_distance_vec"],
                graph_cpu["batch"],
                graph_cpu["batch_size"],
            )
        cpu_times = []
        for _ in range(10):
            t0 = time.perf_counter_ns()
            model_cpu.forward_static(
                graph_cpu["atomic_numbers"],
                graph_cpu["edge_index"],
                graph_cpu["edge_distance"],
                graph_cpu["edge_distance_vec"],
                graph_cpu["batch"],
                graph_cpu["batch_size"],
            )
            t1 = time.perf_counter_ns()
            cpu_times.append((t1 - t0) / 1e6)
    cpu_ms = sum(cpu_times) / len(cpu_times)

    speedup = cpu_ms / avg_ms
    print(f"  CPU avg:     {cpu_ms:.1f} ms")
    print(f"  Neuron avg:  {avg_ms:.1f} ms")
    print(
        f"  Speedup:     {speedup:.2f}x {'(Neuron wins)' if speedup > 1 else '(CPU wins)'}"
    )


if __name__ == "__main__":
    main()
