"""Test EquiformerV3 training on Neuron (eager mode).

Phase 1: Verify forward + backward + optimizer step works in eager mode
on device="neuron" (PyTorch Native Beta 2).

Usage:
    # On CPU (local validation):
    python test_v3_training.py --device cpu

    # On Neuron:
    python test_v3_training.py --device neuron

    # With torch.compile (Phase 2):
    python test_v3_training.py --device neuron --compile

    # BF16 (Phase 3):
    python test_v3_training.py --device neuron --bf16

    # All together:
    python test_v3_training.py --device neuron --compile --bf16
"""

import sys
import os
import types
import contextlib
import time
import argparse
import math

import torch
import torch.nn as nn

# ============================================================
# Stub setup for fairchem dependencies (V3 only needs 3 imports)
# ============================================================

# Create package hierarchy
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

# fairchem.core.common.registry
registry_mod = types.ModuleType("fairchem.core.common.registry")
registry_mod.__package__ = "fairchem.core.common"


class Registry:
    @staticmethod
    def register_model(name):
        def decorator(cls):
            return cls

        return decorator


registry_mod.registry = Registry()
sys.modules["fairchem.core.common.registry"] = registry_mod
sys.modules["fairchem.core.common"].registry = registry_mod

# fairchem.core.common.utils (only conditional_grad needed)
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

# fairchem.core.models.base (GraphModelMixin - minimal version)
base_mod = types.ModuleType("fairchem.core.models.base")
base_mod.__package__ = "fairchem.core.models"


class GraphModelMixin:
    """Minimal stub - forward_static() bypasses generate_graph entirely."""

    def generate_graph(self, data, **kwargs):
        raise NotImplementedError(
            "generate_graph should not be called when using forward_static(). "
            "Use StaticGraphData.get_forward_static_args() instead."
        )


base_mod.GraphModelMixin = GraphModelMixin
sys.modules["fairchem.core.models.base"] = base_mod
sys.modules["fairchem.core.models"].base = base_mod

# Stub torch_geometric (only imported but not actually used in forward path)
try:
    import torch_geometric
except ImportError:
    tg_mod = types.ModuleType("torch_geometric")
    tg_mod.__package__ = "torch_geometric"
    tg_mod.__path__ = []
    tg_utils_mod = types.ModuleType("torch_geometric.utils")
    tg_utils_mod.__package__ = "torch_geometric"
    tg_utils_mod.softmax = None  # commented out in source
    tg_mod.utils = tg_utils_mod
    sys.modules["torch_geometric"] = tg_mod
    sys.modules["torch_geometric.utils"] = tg_utils_mod

# ============================================================
# Add V3 model to path
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try local path (full repo structure) first, then remote (flat) layout
V3_ROOT_LOCAL = os.path.join(SCRIPT_DIR, "equiformer_v3", "experimental", "models")
V3_ROOT_REMOTE = os.path.join(SCRIPT_DIR, "equiformer_v3")

if os.path.isdir(os.path.join(V3_ROOT_LOCAL, "equiformer_v3")):
    V3_ROOT = V3_ROOT_LOCAL
elif os.path.isdir(os.path.join(V3_ROOT_REMOTE, "equiformer_v3")):
    V3_ROOT = V3_ROOT_REMOTE
else:
    raise RuntimeError(
        f"Cannot find equiformer_v3 package. Checked:\n"
        f"  {V3_ROOT_LOCAL}/equiformer_v3\n"
        f"  {V3_ROOT_REMOTE}/equiformer_v3"
    )
sys.path.insert(0, V3_ROOT)

# Now import V3
from equiformer_v3.equiformer_v3 import EquiformerV3_OC
from equiformer_v3.graph_padding import StaticGraphData, create_random_graph


def create_model(device, dtype=torch.float32):
    """Create a small V3 model for training tests."""
    model = EquiformerV3_OC(
        direct_prediction=True,
        regress_forces=True,
        regress_stress=False,
        num_layers=4,
        num_channels=128,
        lmax=4,
        mmax=2,
        num_radial_basis=64,
        max_neighbors=20,
        max_radius=12.0,
        num_heads=8,
        attn_hidden_channels=64,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        use_grid_mlp=True,
        use_envelope=True,
        # Disable all dropout for compile compatibility
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        # Other settings
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
        gradient_checkpointing_block_list=None,
    )
    model = model.to(device=device, dtype=dtype)
    return model


def create_batch(
    num_atoms=50, max_neighbors=20, batch_size=2, device="cpu", dtype=torch.float32
):
    """Create a random training batch.

    Graph construction (random positions, neighbor finding) is done on CPU,
    then tensors are moved to target device. This avoids unsupported ops on Neuron.
    """
    # Always build graph on CPU (some ops like repeat_interleave aren't on Neuron)
    graph = create_random_graph(
        num_atoms=num_atoms,
        max_neighbors=max_neighbors,
        max_radius=12.0,
        batch_size=batch_size,
        device="cpu",
        dtype=dtype,
    )
    # Move tensors to target device
    if device != "cpu":
        graph["atomic_numbers"] = graph["atomic_numbers"].to(device)
        graph["edge_index"] = graph["edge_index"].to(device)
        graph["edge_distance"] = graph["edge_distance"].to(device)
        graph["edge_distance_vec"] = graph["edge_distance_vec"].to(device)
        graph["batch"] = graph["batch"].to(device)
    return graph


def train_step_eager(model, graph, optimizer, force_weight=10.0):
    """Single training step in eager mode.

    Loss = MSE(energy) + force_weight * MSE(forces)
    """
    optimizer.zero_grad()

    outputs = model.forward_static(
        graph["atomic_numbers"],
        graph["edge_index"],
        graph["edge_distance"],
        graph["edge_distance_vec"],
        graph["batch"],
        graph["batch_size"],
    )

    # Fake targets (random for testing)
    energy_target = torch.randn(
        graph["batch_size"],
        device=outputs["energy"].device,
        dtype=outputs["energy"].dtype,
    )
    forces_target = torch.randn_like(outputs["forces"])

    # Loss (use manual MSE instead of F.mse_loss -- Ticket 40: F.mse_loss backward
    # fails with torch.compile on Neuron due to torch.constant.int legalization)
    energy_loss = ((outputs["energy"] - energy_target) ** 2).mean()
    forces_loss = ((outputs["forces"] - forces_target) ** 2).mean()
    loss = energy_loss + force_weight * forces_loss

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "energy_loss": energy_loss.item(),
        "forces_loss": forces_loss.item(),
    }


def train_step_compiled(compiled_fn, graph, optimizer, force_weight=10.0):
    """Single training step using a compiled forward function."""
    optimizer.zero_grad()

    outputs = compiled_fn(
        graph["atomic_numbers"],
        graph["edge_index"],
        graph["edge_distance"],
        graph["edge_distance_vec"],
        graph["batch"],
        graph["batch_size"],
    )

    energy_target = torch.randn(
        graph["batch_size"],
        device=outputs["energy"].device,
        dtype=outputs["energy"].dtype,
    )
    forces_target = torch.randn_like(outputs["forces"])

    energy_loss = ((outputs["energy"] - energy_target) ** 2).mean()
    forces_loss = ((outputs["forces"] - forces_target) ** 2).mean()
    loss = energy_loss + force_weight * forces_loss

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "energy_loss": energy_loss.item(),
        "forces_loss": forces_loss.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Test EquiformerV3 training on Neuron")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "neuron"],
        help="Device to run on",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile(backend='neuron')"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument(
        "--num-atoms", type=int, default=50, help="Number of atoms per molecule"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Number of molecules per batch"
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"=" * 60)
    print(f"EquiformerV3 Training Test")
    print(f"=" * 60)
    print(f"  Device:     {device}")
    print(f"  Compile:    {args.compile}")
    print(f"  Dtype:      {dtype}")
    print(f"  Atoms:      {args.num_atoms}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps:      {args.steps}")
    print(f"  LR:         {args.lr}")
    print()

    # Create model
    print("Creating model...")
    t0 = time.time()
    model = create_model(device=device, dtype=dtype)
    print(f"  Model created in {time.time() - t0:.2f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create data
    print("Creating training batch...")
    graph = create_batch(
        num_atoms=args.num_atoms,
        max_neighbors=20,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
    )
    print(f"  Atoms: {graph['atomic_numbers'].shape[0]}")
    print(f"  Edges: {graph['edge_index'].shape[1]}")
    print()

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Compile if requested
    if args.compile:
        print("Compiling model with torch.compile(backend='neuron')...")
        t0 = time.time()
        compiled_fn = torch.compile(model.forward_static, backend="neuron")
        print(f"  torch.compile setup in {time.time() - t0:.2f}s")
        print()

    # Training loop
    print(f"Running {args.steps} training steps...")
    print("-" * 60)

    model.train()
    times = []

    for step in range(args.steps):
        t0 = time.time()

        if args.compile:
            metrics = train_step_compiled(compiled_fn, graph, optimizer)
        else:
            metrics = train_step_eager(model, graph, optimizer)

        step_time = time.time() - t0
        times.append(step_time)

        print(
            f"  Step {step + 1}/{args.steps}: "
            f"loss={metrics['loss']:.6f} "
            f"(E={metrics['energy_loss']:.6f}, F={metrics['forces_loss']:.6f}) "
            f"time={step_time:.3f}s"
        )

    print("-" * 60)
    print()

    # Summary
    if len(times) > 1:
        # Skip first step (compilation/warmup)
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"Summary:")
        print(f"  First step (warmup):   {times[0]:.3f}s")
        print(f"  Avg step (after warm): {avg_time:.3f}s")
        print(f"  Total time:            {sum(times):.3f}s")
    else:
        print(f"Summary:")
        print(f"  Step time: {times[0]:.3f}s")

    # Verify gradients exist
    print()
    print("Gradient check:")
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad += 1
        else:
            no_grad += 1
    print(f"  Parameters with gradients: {has_grad}")
    print(f"  Parameters without gradients: {no_grad}")

    if no_grad > 0:
        print("  WARNING: Some parameters have no gradient!")
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"    - {name}")
                if has_grad > 10:
                    print(f"    ... (showing first few)")
                    break

    print()
    print("PASSED" if has_grad > 0 else "FAILED")


if __name__ == "__main__":
    main()
