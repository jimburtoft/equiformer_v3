"""Per-layer compilation approach: compile each component separately.

Strategy:
- Compile embedding as one NEFF
- Compile each transformer block as a separate NEFF (7 NEFFs)
- Compile output head as one NEFF

This should give:
- Small NEFFs (minimal spilling)
- Reduced compile time (smaller graphs)
- Clear separation of concerns
"""

import sys, os, types, contextlib, time
import torch
import torch._dynamo

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

# Production config
PRODUCTION_CONFIG = dict(
    direct_prediction=True,
    regress_forces=True,
    regress_stress=False,
    num_layers=7,
    num_channels=128,
    lmax=4,
    mmax=4,
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
)

print("=" * 60)
print("PER-LAYER COMPILATION APPROACH")
print("=" * 60)

graph = create_random_graph(
    num_atoms=100,
    max_neighbors=10,
    max_radius=12.0,
    batch_size=1,
    device="cpu",
    dtype=torch.float32,
)
edges = graph["edge_index"].shape[1]
print(f"Edges: {edges}")

model = EquiformerV3_OC(**PRODUCTION_CONFIG).eval()
model.build_fused_linear()
model.pre_expand_weights()
model = model.to("neuron")
data = {
    k: v.to("neuron") if isinstance(v, torch.Tensor) else v for k, v in graph.items()
}

# Setup compilation infrastructure
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch_neuronx.neuron_dynamo_backend.backend import (
    preprocess_graph,
    get_compile_decomposition_table,
    make_compiler,
)

decomposition_table = get_compile_decomposition_table()


def _fw_compiler(gm, example_inputs):
    gm_processed, analysis = preprocess_graph(gm)
    compiler = make_compiler(analysis, options=None)
    return compiler(gm_processed, example_inputs)


def _compile_backend(gm, example_inputs):
    aot_backend = aot_autograd(
        fw_compiler=_fw_compiler,
        bw_compiler=make_boxed_func,
        keep_inference_input_mutations=True,
        decompositions=decomposition_table,
    )
    return aot_backend(gm, example_inputs)


try:
    register_backend(name="neuron_aot_perlayer", compiler_fn=_compile_backend)
except Exception:
    pass

# ============================================================
# Step 1: Compile embedding
# ============================================================
print("\n--- Step 1: Compiling embedding ---")


# Set model attributes that forward_static normally sets
model.batch_size = data["batch_size"]
model.dtype = data["edge_distance_vec"].dtype
model.device = data["edge_distance_vec"].device


def embedding_fn(atomic_numbers, edge_index, edge_distance, edge_distance_vec):
    source_atomic_numbers = atomic_numbers[edge_index[0]]
    target_atomic_numbers = atomic_numbers[edge_index[1]]
    edge_distance_expanded, edge_envelope_weight = model._forward_edge(
        edge_distance, edge_distance_vec
    )
    x = model._forward_embedding(
        atomic_numbers, edge_distance_expanded, edge_index, edge_envelope_weight
    )
    return (
        x,
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance_expanded,
        edge_envelope_weight,
    )


torch._dynamo.reset()
t0 = time.time()
compiled_embedding = torch.compile(embedding_fn, backend="neuron_aot_perlayer")
with torch.no_grad():
    x, src_an, tgt_an, edge_dist_exp, edge_env_w = compiled_embedding(
        data["atomic_numbers"],
        data["edge_index"],
        data["edge_distance"],
        data["edge_distance_vec"],
    )
embed_compile_time = time.time() - t0
print(f"  Embedding compiled in {embed_compile_time:.1f}s")
print(f"  x shape: {x.shape}")

# ============================================================
# Step 2: Compile each transformer block separately
# ============================================================
print("\n--- Step 2: Compiling transformer blocks ---")

compiled_blocks = []
block_compile_times = []

for i in range(model.num_layers):
    block_i = model.blocks[i]

    # Create a closure that captures block_i
    def make_block_fn(b):
        def fn(
            x,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
        ):
            return b(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight,
                batch=None,
            )

        return fn

    torch._dynamo.reset()
    t0 = time.time()
    compiled_i = torch.compile(make_block_fn(block_i), backend="neuron_aot_perlayer")
    with torch.no_grad():
        x_out = compiled_i(
            x, src_an, tgt_an, edge_dist_exp, data["edge_index"], edge_env_w
        )
    t_i = time.time() - t0
    compiled_blocks.append(compiled_i)
    block_compile_times.append(t_i)
    print(f"  Block {i} compiled in {t_i:.1f}s")

# ============================================================
# Step 3: Compile output head
# ============================================================
print("\n--- Step 3: Compiling output head ---")


def output_fn(
    x,
    source_atomic_numbers,
    target_atomic_numbers,
    edge_distance,
    edge_index,
    edge_envelope_weight,
    batch,
):
    x = model.norm(x)
    x_scalar, _ = torch.split(x, [1, x.shape[1] - 1], dim=1)
    x_scalar = x_scalar.view(x_scalar.shape[0], model.num_channels)

    node_energy = model.energy_block(x_scalar)
    energy = scatter_add_plain(node_energy.view(-1), batch, 1)
    energy = energy / model.avg_num_nodes

    forces = model.force_block(
        x,
        source_atomic_numbers,
        target_atomic_numbers,
        edge_distance,
        edge_index,
        edge_envelope_weight,
    )
    _, forces, _ = torch.split(forces, [1, 3, forces.shape[1] - 4], dim=1)
    forces = forces.view(-1, 3)
    return energy, forces


# Run all blocks to get final x for output compilation
with torch.no_grad():
    x_final = x
    for i in range(model.num_layers):
        x_final = compiled_blocks[i](
            x_final, src_an, tgt_an, edge_dist_exp, data["edge_index"], edge_env_w
        )

torch._dynamo.reset()
t0 = time.time()
compiled_output = torch.compile(output_fn, backend="neuron_aot_perlayer")
with torch.no_grad():
    energy, forces = compiled_output(
        x_final,
        src_an,
        tgt_an,
        edge_dist_exp,
        data["edge_index"],
        edge_env_w,
        data["batch"],
    )
output_compile_time = time.time() - t0
print(f"  Output head compiled in {output_compile_time:.1f}s")
print(f"  energy: {energy.cpu().item():.6f}, forces: {forces.shape}")

# ============================================================
# Step 4: Benchmark
# ============================================================
print("\n--- Step 4: Benchmarking ---")


def full_forward_per_layer():
    with torch.no_grad():
        x, src_an, tgt_an, edge_dist_exp, edge_env_w = compiled_embedding(
            data["atomic_numbers"],
            data["edge_index"],
            data["edge_distance"],
            data["edge_distance_vec"],
        )
        for i in range(model.num_layers):
            x = compiled_blocks[i](
                x, src_an, tgt_an, edge_dist_exp, data["edge_index"], edge_env_w
            )
        energy, forces = compiled_output(
            x,
            src_an,
            tgt_an,
            edge_dist_exp,
            data["edge_index"],
            edge_env_w,
            data["batch"],
        )
    return energy, forces


# Warmup
print("  Warmup (5 iters)...")
for _ in range(5):
    full_forward_per_layer()

# Benchmark
print("  Benchmarking (20 iters)...")
times = []
for _ in range(20):
    t0 = time.perf_counter()
    energy, forces = full_forward_per_layer()
    times.append(time.perf_counter() - t0)

avg_ms = sum(times) / len(times) * 1000
min_ms = min(times) * 1000

total_compile = embed_compile_time + sum(block_compile_times) + output_compile_time

print(f"\n{'=' * 60}")
print(f"RESULTS: Per-layer compiled (N=100, K=10, {edges} edges)")
print(f"{'=' * 60}")
print(f"  Latency: {avg_ms:.2f} ms avg, {min_ms:.2f} ms min")
print(f"  Total compile time: {total_compile:.1f}s")
print(f"    Embedding: {embed_compile_time:.1f}s")
print(
    f"    Blocks: {sum(block_compile_times):.1f}s ({min(block_compile_times):.1f}-{max(block_compile_times):.1f}s each)"
)
print(f"    Output: {output_compile_time:.1f}s")
print(f"\n  COMPARISON:")
print(f"    Eager:                       757 ms")
print(f"    Monolithic compiled (9 brk): 232 ms (3.3x vs eager)")
print(
    f"    Per-layer compiled:          {avg_ms:.2f} ms ({757 / avg_ms:.1f}x vs eager)"
)
if avg_ms < 232:
    print(f"    Improvement over monolithic: {232 / avg_ms:.2f}x")
else:
    print(f"    Regression vs monolithic: {avg_ms / 232:.2f}x slower")
