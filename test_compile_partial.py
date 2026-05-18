"""Test torch.compile with fullgraph=False (allow graph breaks)."""
import sys, os, types, contextlib, time
import torch

# ============================================================
# Stubs for fairchem
# ============================================================
for pkg_name in [
    "fairchem", "fairchem.core", "fairchem.core.common", "fairchem.core.models",
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

print("Creating model...", flush=True)
model = EquiformerV3_OC(
    direct_prediction=True, regress_forces=True, regress_stress=False,
    num_layers=2, num_channels=64, lmax=2, mmax=2,
    num_radial_basis=32, max_neighbors=10, max_radius=12.0,
    num_heads=4, attn_hidden_channels=32, attn_alpha_channels=16, attn_value_channels=8,
    ffn_hidden_channels=64, use_grid_mlp=True, use_envelope=True,
    alpha_drop=0.0, attn_mask_rate=0.0, attn_weights_drop=0.0, value_drop=0.0,
    drop_path_rate=0.0, proj_drop=0.0, ffn_drop=0.0,
    use_pbc=False, otf_graph=False, use_atom_edge_embedding=True,
    use_attn_renorm=True, use_add_merge=False, use_rad_l_parametrization=True,
    norm_type="merge_layer_norm",
    attn_activation="sep-merge_gates2_swiglu",
    ffn_activation="sep-merge_gates2_swiglu",
    use_gate_force_head=True,
).to(device="neuron", dtype=torch.float32)
model.train()

print("Creating data...", flush=True)
graph = create_random_graph(num_atoms=50, max_neighbors=10, max_radius=12.0, batch_size=2, device="cpu", dtype=torch.float32)
data = {k: v.to("neuron") if isinstance(v, torch.Tensor) else v for k, v in graph.items()}

print("Compiling with fullgraph=False...", flush=True)
t0 = time.time()
compiled_fn = torch.compile(model.forward_static, backend="neuron", fullgraph=False)
print(f"  torch.compile setup: {time.time()-t0:.2f}s", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(5):
    t0 = time.time()
    optimizer.zero_grad()
    outputs = compiled_fn(
        data["atomic_numbers"], data["edge_index"],
        data["edge_distance"], data["edge_distance_vec"],
        data["batch"], data["batch_size"],
    )
    energy_target = torch.zeros(data["batch_size"], device="neuron", dtype=torch.float32)
    forces_target = torch.zeros_like(outputs["forces"])
    loss = ((outputs["energy"] - energy_target)**2).mean() + 10.0 * ((outputs["forces"] - forces_target)**2).mean()
    loss.backward()
    optimizer.step()
    print(f"  Step {step+1}: loss={loss.cpu().item():.6f} time={time.time()-t0:.1f}s", flush=True)

print("DONE - compile with graph breaks", flush=True)
