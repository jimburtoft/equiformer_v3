"""Benchmark: Full production-size EquiformerV3 compiled training.

Model config: 7 layers, lmax=4, mmax=4, C=128 (MPtrj-equivalent)
This is ~10x larger than the 2-layer test model.

Usage (in Docker container on trn2):
    python benchmark_large_model.py
"""

import sys, os, types, contextlib, time
import torch

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

# Backend setup
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch_neuronx.neuron_dynamo_backend.backend import (
    preprocess_graph,
    get_compile_decomposition_table,
    make_compiler,
)

decomposition_table = get_compile_decomposition_table()


def replace_view_with_reshape(gm):
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.view.default:
            node.target = torch.ops.aten.reshape.default
    gm.recompile()
    return gm


bwd_compile_count = [0]


def bwd_compiler(gm, example_inputs):
    gm = replace_view_with_reshape(gm)
    bwd_compile_count[0] += 1
    n_ops = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    print(f"  [BWD #{bwd_compile_count[0]}] {n_ops} ops, compiling with neuron...")
    try:
        gm_processed, analysis = preprocess_graph(gm)
        compiler = make_compiler(analysis, options=None)
        compiled = compiler(gm_processed, example_inputs)
        print(f"  [BWD #{bwd_compile_count[0]}] SUCCESS")
        return compiled
    except Exception as e:
        print(f"  [BWD #{bwd_compile_count[0]}] FAILED: {str(e)[:200]}")
        print(f"  Falling back to eager for this subgraph")
        return make_boxed_func(gm.forward)


def compile_backend(gm, example_inputs):
    gm, analysis_results = preprocess_graph(gm)
    fw_compiler = make_compiler(analysis_results, options=None)
    aot_backend = aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bwd_compiler,
        keep_inference_input_mutations=True,
        decompositions=decomposition_table,
    )
    return aot_backend(gm, example_inputs)


register_backend(name="compile_both", compiler_fn=compile_backend)

# ============================================================
# LARGE MODEL CONFIG (production-size)
# ============================================================
print("=" * 60)
print("BENCHMARK: Production-size EquiformerV3")
print("  7 layers, lmax=4, mmax=4, C=128")
print("=" * 60)

print("\nCreating model...")
t_model = time.time()
model = EquiformerV3_OC(
    direct_prediction=True,
    regress_forces=True,
    regress_stress=False,
    num_layers=7,  # Production: 7 layers
    num_channels=128,  # Production: 128 channels
    lmax=4,  # Production: lmax=4
    mmax=4,  # Production: mmax=4
    num_radial_basis=32,
    max_neighbors=10,
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
    use_add_merge=False,
    use_rad_l_parametrization=True,
    norm_type="merge_layer_norm",
    attn_activation="sep-merge_gates2_swiglu",
    ffn_activation="sep-merge_gates2_swiglu",
    use_gate_force_head=True,
).to(device="neuron", dtype=torch.float32)
model.train()

num_params = sum(p.numel() for p in model.parameters())
print(f"  Model created in {time.time() - t_model:.1f}s")
print(f"  Parameters: {num_params:,} ({num_params * 4 / 1e6:.1f} MB FP32)")

print("\nCreating test data (50 atoms, batch=2)...")
graph = create_random_graph(
    num_atoms=50,
    max_neighbors=10,
    max_radius=12.0,
    batch_size=2,
    device="cpu",
    dtype=torch.float32,
)
data = {
    k: v.to("neuron") if isinstance(v, torch.Tensor) else v for k, v in graph.items()
}

# First try eager to get baseline
print("\n--- Eager baseline (3 steps) ---")
optimizer_eager = torch.optim.AdamW(model.parameters(), lr=1e-4)
eager_times = []
for step in range(3):
    optimizer_eager.zero_grad()
    t0 = time.time()
    outputs = model.forward_static(
        data["atomic_numbers"],
        data["edge_index"],
        data["edge_distance"],
        data["edge_distance_vec"],
        data["batch"],
        data["batch_size"],
    )
    energy_target = torch.zeros(
        data["batch_size"], device="neuron", dtype=torch.float32
    )
    forces_target = torch.zeros_like(outputs["forces"])
    loss = ((outputs["energy"] - energy_target) ** 2).mean() + 10.0 * (
        (outputs["forces"] - forces_target) ** 2
    ).mean()
    loss.backward()
    optimizer_eager.step()
    elapsed = time.time() - t0
    eager_times.append(elapsed)
    print(f"  Eager step {step}: {elapsed:.2f}s, loss={loss.cpu().item():.6f}")

eager_avg = sum(eager_times[1:]) / len(eager_times[1:])
print(f"  Eager avg (steps 1-2): {eager_avg:.2f}s")

# Now try compiled
print("\n--- Compiled training (5 steps) ---")
# Reset model state for fair comparison
torch._dynamo.reset()
compiled_fn = torch.compile(model.forward_static, backend="compile_both")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
compiled_times = []

for step in range(5):
    optimizer.zero_grad()
    t0 = time.time()
    outputs = compiled_fn(
        data["atomic_numbers"],
        data["edge_index"],
        data["edge_distance"],
        data["edge_distance_vec"],
        data["batch"],
        data["batch_size"],
    )
    energy_target = torch.zeros(
        data["batch_size"], device="neuron", dtype=torch.float32
    )
    forces_target = torch.zeros_like(outputs["forces"])
    loss = ((outputs["energy"] - energy_target) ** 2).mean() + 10.0 * (
        (outputs["forces"] - forces_target) ** 2
    ).mean()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    compiled_times.append(elapsed)
    print(f"  Compiled step {step}: {elapsed:.2f}s, loss={loss.cpu().item():.6f}")

compiled_avg = sum(compiled_times[1:]) / len(compiled_times[1:])

print(f"\n{'=' * 60}")
print(f"RESULTS: Production EquiformerV3 (7L, L4, C128)")
print(f"{'=' * 60}")
print(f"  Model size: {num_params:,} params ({num_params * 4 / 1e6:.1f} MB)")
print(f"  Eager avg: {eager_avg:.2f} s/step")
print(f"  Compiled warmup: {compiled_times[0]:.1f}s")
print(f"  Compiled avg (post-warmup): {compiled_avg:.2f} s/step")
print(f"  Speedup: {eager_avg / compiled_avg:.1f}x")
print(
    f"  Grad count: {sum(1 for p in model.parameters() if p.grad is not None)}/{sum(1 for _ in model.parameters())}"
)
