"""Comprehensive EquiformerV3 training benchmark on Neuron.

Tests:
  1. Eager FP32 training (baseline)
  2. torch.compile FP32 training
  3. Eager BF16 training
  4. torch.compile BF16 training

Measures: step time, loss convergence, gradient health.
"""

import sys, os, types, contextlib, time, argparse
import torch

# ============================================================
# Stubs for fairchem
# ============================================================
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

try:
    import torch_geometric
except ImportError:
    tg_mod = types.ModuleType("torch_geometric")
    tg_mod.__package__ = "torch_geometric"
    tg_mod.__path__ = []
    tg_utils_mod = types.ModuleType("torch_geometric.utils")
    tg_utils_mod.__package__ = "torch_geometric"
    tg_utils_mod.softmax = None
    tg_mod.utils = tg_utils_mod
    sys.modules["torch_geometric"] = tg_mod
    sys.modules["torch_geometric.utils"] = tg_utils_mod

# ============================================================
# Model imports
# ============================================================
sys.path.insert(0, os.path.expanduser("~/equiformer_v3"))
from equiformer_v3.equiformer_v3 import EquiformerV3_OC
from equiformer_v3.graph_padding import create_random_graph


def create_model(dtype=torch.float32):
    """Create V3 model with test config."""
    model = EquiformerV3_OC(
        direct_prediction=True,
        regress_forces=True,
        regress_stress=False,
        num_layers=2,
        num_channels=64,
        lmax=2,
        mmax=2,
        num_radial_basis=32,
        max_neighbors=10,
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
        use_add_merge=False,
        use_rad_l_parametrization=True,
        norm_type="merge_layer_norm",
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        use_gate_force_head=True,
    )
    model = model.to(device="neuron", dtype=dtype)
    return model


def create_data(num_atoms=50, batch_size=2, dtype=torch.float32):
    """Create graph data on CPU, move to neuron."""
    graph = create_random_graph(
        num_atoms=num_atoms,
        max_neighbors=10,
        max_radius=12.0,
        batch_size=batch_size,
        device="cpu",
        dtype=dtype,
    )
    return {
        "atomic_numbers": graph["atomic_numbers"].to("neuron"),
        "edge_index": graph["edge_index"].to("neuron"),
        "edge_distance": graph["edge_distance"].to("neuron"),
        "edge_distance_vec": graph["edge_distance_vec"].to("neuron"),
        "batch": graph["batch"].to("neuron"),
        "batch_size": graph["batch_size"],
    }


def train_step(model_fn, data, optimizer):
    """Single training step. Returns loss value and step time."""
    t0 = time.time()
    optimizer.zero_grad()

    outputs = model_fn(
        data["atomic_numbers"],
        data["edge_index"],
        data["edge_distance"],
        data["edge_distance_vec"],
        data["batch"],
        data["batch_size"],
    )

    # Manual MSE (Ticket 40: F.mse_loss backward fails on Neuron)
    energy_target = torch.zeros(
        data["batch_size"], device="neuron", dtype=outputs["energy"].dtype
    )
    forces_target = torch.zeros_like(outputs["forces"])
    energy_loss = ((outputs["energy"] - energy_target) ** 2).mean()
    forces_loss = ((outputs["forces"] - forces_target) ** 2).mean()
    loss = energy_loss + 10.0 * forces_loss

    loss.backward()
    optimizer.step()

    return loss.cpu().item(), time.time() - t0


def benchmark_mode(mode_name, model, data, num_steps=5, use_compile=False):
    """Run training steps and collect metrics."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Mode: {mode_name}", flush=True)
    print(f"{'=' * 60}", flush=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if use_compile:
        print("  Compiling with torch.compile(backend='neuron')...", flush=True)
        t0 = time.time()
        model_fn = torch.compile(model.forward_static, backend="neuron")
        print(f"  Compile setup: {time.time() - t0:.2f}s", flush=True)
    else:
        model_fn = model.forward_static

    losses = []
    times = []

    for step in range(num_steps):
        loss_val, step_time = train_step(model_fn, data, optimizer)
        losses.append(loss_val)
        times.append(step_time)
        print(
            f"  Step {step + 1}/{num_steps}: loss={loss_val:.6f} time={step_time:.1f}s",
            flush=True,
        )

    # Stats
    warmup_time = times[0]
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    print(f"\n  Results:", flush=True)
    print(f"    Warmup (step 1):  {warmup_time:.1f}s", flush=True)
    print(f"    Avg (steps 2+):   {avg_time:.2f}s", flush=True)
    print(f"    Loss start->end:  {losses[0]:.6f} -> {losses[-1]:.6f}", flush=True)
    print(f"    Gradients:        {has_grad}/{total_params}", flush=True)
    print(
        f"    Status:           {'PASS' if has_grad == total_params and losses[-1] < losses[0] else 'CHECK'}",
        flush=True,
    )

    return {
        "mode": mode_name,
        "warmup_s": warmup_time,
        "avg_step_s": avg_time,
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "grads": has_grad,
        "total_params": total_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-atoms", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--modes",
        type=str,
        default="eager_fp32,compile_fp32,eager_bf16,compile_bf16",
        help="Comma-separated modes to test",
    )
    args = parser.parse_args()

    modes_to_run = args.modes.split(",")

    print("=" * 60, flush=True)
    print("EquiformerV3 Training Benchmark", flush=True)
    print("=" * 60, flush=True)
    print(
        f"  Atoms: {args.num_atoms}, Batch: {args.batch_size}, Steps: {args.steps}",
        flush=True,
    )
    print(f"  Modes: {modes_to_run}", flush=True)
    print(f"  PyTorch: {torch.__version__}", flush=True)

    results = []

    for mode in modes_to_run:
        use_compile = "compile" in mode
        use_bf16 = "bf16" in mode
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        # Fresh model for each mode
        model = create_model(dtype=dtype)
        data = create_data(
            num_atoms=args.num_atoms, batch_size=args.batch_size, dtype=dtype
        )

        result = benchmark_mode(
            mode_name=mode,
            model=model,
            data=data,
            num_steps=args.steps,
            use_compile=use_compile,
        )
        results.append(result)

        # Clean up
        del model, data
        torch.neuron.synchronize() if hasattr(torch, "neuron") else None

    # Summary table
    print(f"\n\n{'=' * 60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(
        f"{'Mode':<20} {'Warmup':>8} {'Avg Step':>10} {'Loss End':>10} {'Grads':>8}",
        flush=True,
    )
    print("-" * 60, flush=True)
    for r in results:
        status = "OK" if r["grads"] == r["total_params"] else "FAIL"
        print(
            f"{r['mode']:<20} {r['warmup_s']:>7.1f}s {r['avg_step_s']:>9.2f}s {r['loss_end']:>10.6f} {r['grads']}/{r['total_params']} {status}",
            flush=True,
        )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
