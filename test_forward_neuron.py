"""Minimal test: forward pass on Neuron device."""

import sys, os, types, contextlib, traceback
import torch

# Setup stubs
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

sys.path.insert(0, os.path.expanduser("~/equiformer_v3"))
from equiformer_v3.equiformer_v3 import EquiformerV3_OC
from equiformer_v3.graph_padding import create_random_graph

print("1. Creating model on CPU...", flush=True)
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
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

print("2. Creating graph on CPU...", flush=True)
graph = create_random_graph(
    num_atoms=20,
    max_neighbors=10,
    max_radius=12.0,
    batch_size=1,
    device="cpu",
    dtype=torch.float32,
)
an = graph["atomic_numbers"]
ei = graph["edge_index"]
ed = graph["edge_distance"]
edv = graph["edge_distance_vec"]
bt = graph["batch"]
bs = graph["batch_size"]
print(f"   Atoms: {an.shape[0]}, Edges: {ei.shape[1]}", flush=True)

print("3. CPU forward...", flush=True)
model.eval()
with torch.no_grad():
    out_cpu = model.forward_static(an, ei, ed, edv, bt, bs)
print(f"   Energy: {out_cpu['energy'].item():.6f}", flush=True)
print(f"   Forces: {out_cpu['forces'].shape}", flush=True)

print("4. Moving model to neuron...", flush=True)
model = model.to("neuron")
print("   Done.", flush=True)

print("5. Moving data to neuron...", flush=True)
an_n = an.to("neuron")
ei_n = ei.to("neuron")
ed_n = ed.to("neuron")
edv_n = edv.to("neuron")
bt_n = bt.to("neuron")
print("   Done.", flush=True)

print("6. Neuron forward (inference)...", flush=True)
try:
    with torch.no_grad():
        out_n = model.forward_static(an_n, ei_n, ed_n, edv_n, bt_n, bs)
    print(f"   Energy: {out_n['energy'].cpu().item():.6f}", flush=True)
    print(f"   Forces: {out_n['forces'].shape}", flush=True)
    print("FORWARD ON NEURON OK!", flush=True)
except Exception as e:
    traceback.print_exc()
    print(f"FAILED: {e}", flush=True)
    sys.exit(1)

print("\n7. Training step (eager)...", flush=True)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()

try:
    outputs = model.forward_static(an_n, ei_n, ed_n, edv_n, bt_n, bs)
    energy_target = torch.randn(bs, device="neuron")
    forces_target = torch.randn_like(outputs["forces"])
    loss = torch.nn.functional.mse_loss(
        outputs["energy"], energy_target
    ) + 10.0 * torch.nn.functional.mse_loss(outputs["forces"], forces_target)
    print(f"   Loss: {loss.cpu().item():.6f}", flush=True)
    loss.backward()
    optimizer.step()

    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"   Grads: {has_grad}/{total}", flush=True)
    print("TRAINING STEP OK!", flush=True)
except Exception as e:
    traceback.print_exc()
    print(f"TRAINING FAILED: {e}", flush=True)
    sys.exit(1)
