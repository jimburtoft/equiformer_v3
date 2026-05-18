"""Test multi-core data parallelism with eager training.
Runs 4 independent workers on separate NeuronCores (LNC=2 = 4 cores).
"""
import sys, os, types, contextlib, time
import torch
import multiprocessing as mp

# ============================================================
# Stubs
# ============================================================
def setup_stubs():
    for pkg_name in ["fairchem", "fairchem.core", "fairchem.core.common", "fairchem.core.models"]:
        mod = types.ModuleType(pkg_name); mod.__package__ = pkg_name; mod.__path__ = []
        sys.modules[pkg_name] = mod
        parts = pkg_name.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            if parent_name in sys.modules: setattr(sys.modules[parent_name], parts[-1], mod)

    registry_mod = types.ModuleType("fairchem.core.common.registry"); registry_mod.__package__ = "fairchem.core.common"
    class Registry:
        @staticmethod
        def register_model(name): return lambda cls: cls
    registry_mod.registry = Registry()
    sys.modules["fairchem.core.common.registry"] = registry_mod
    sys.modules["fairchem.core.common"].registry = registry_mod

    utils_mod = types.ModuleType("fairchem.core.common.utils"); utils_mod.__package__ = "fairchem.core.common"
    @contextlib.contextmanager
    def conditional_grad(flag):
        if flag: yield
        else:
            with torch.no_grad(): yield
    utils_mod.conditional_grad = conditional_grad
    sys.modules["fairchem.core.common.utils"] = utils_mod
    sys.modules["fairchem.core.common"].utils = utils_mod

    base_mod = types.ModuleType("fairchem.core.models.base"); base_mod.__package__ = "fairchem.core.models"
    class GraphModelMixin:
        def generate_graph(self, data, **kwargs): raise NotImplementedError
    base_mod.GraphModelMixin = GraphModelMixin
    sys.modules["fairchem.core.models.base"] = base_mod
    sys.modules["fairchem.core.models"].base = base_mod

    try: import torch_geometric
    except ImportError:
        tg_mod = types.ModuleType("torch_geometric"); tg_mod.__package__ = "torch_geometric"; tg_mod.__path__ = []
        tg_utils_mod = types.ModuleType("torch_geometric.utils"); tg_utils_mod.__package__ = "torch_geometric"; tg_utils_mod.softmax = None
        tg_mod.utils = tg_utils_mod
        sys.modules["torch_geometric"] = tg_mod; sys.modules["torch_geometric.utils"] = tg_utils_mod


def worker(core_id, num_steps, result_queue):
    """Worker that trains on a single NeuronCore."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    setup_stubs()
    
    sys.path.insert(0, os.path.expanduser("~/equiformer_v3"))
    from equiformer_v3.equiformer_v3 import EquiformerV3_OC
    from equiformer_v3.graph_padding import create_random_graph

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
    
    graph = create_random_graph(num_atoms=50, max_neighbors=10, max_radius=12.0, batch_size=2, device="cpu", dtype=torch.float32)
    data = {k: v.to("neuron") if isinstance(v, torch.Tensor) else v for k, v in graph.items()}
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    times = []
    losses = []
    for step in range(num_steps):
        t0 = time.time()
        optimizer.zero_grad()
        outputs = model.forward_static(
            data["atomic_numbers"], data["edge_index"],
            data["edge_distance"], data["edge_distance_vec"],
            data["batch"], data["batch_size"],
        )
        energy_target = torch.zeros(data["batch_size"], device="neuron", dtype=torch.float32)
        forces_target = torch.zeros_like(outputs["forces"])
        loss = ((outputs["energy"] - energy_target)**2).mean() + 10.0 * ((outputs["forces"] - forces_target)**2).mean()
        loss.backward()
        optimizer.step()
        elapsed = time.time() - t0
        times.append(elapsed)
        losses.append(loss.cpu().item())
    
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    result_queue.put({
        "core": core_id,
        "warmup": times[0],
        "avg_step": avg_time,
        "loss_start": losses[0],
        "loss_end": losses[-1],
    })


if __name__ == "__main__":
    mp.set_start_method("spawn")
    num_cores = 4  # LNC=2 on trn2.3xlarge
    num_steps = 5
    
    print("=" * 60, flush=True)
    print(f"Multi-Core Data Parallelism Test (DP={num_cores})", flush=True)
    print("=" * 60, flush=True)
    print(f"  Cores: {num_cores}, Steps: {num_steps}", flush=True)
    
    result_queue = mp.Queue()
    
    t_total = time.time()
    processes = []
    for core_id in range(num_cores):
        p = mp.Process(target=worker, args=(core_id, num_steps, result_queue))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    total_time = time.time() - t_total
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda x: x["core"])
    
    print(f"\n{'Core':<6} {'Warmup':>8} {'Avg Step':>10} {'Loss End':>10}", flush=True)
    print("-" * 40, flush=True)
    for r in results:
        print(f"{r['core']:<6} {r['warmup']:>7.1f}s {r['avg_step']:>9.2f}s {r['loss_end']:>10.6f}", flush=True)
    
    # Aggregate throughput
    if results:
        avg_step_all = sum(r["avg_step"] for r in results) / len(results)
        effective_throughput = num_cores / avg_step_all  # steps per second across all cores
        single_core_throughput = 1.0 / results[0]["avg_step"]
        print(f"\n  Total wall time: {total_time:.1f}s", flush=True)
        print(f"  Avg step time (per core): {avg_step_all:.2f}s", flush=True)
        print(f"  Effective throughput: {effective_throughput:.2f} graphs/s (DP={num_cores})", flush=True)
        print(f"  Single-core throughput: {single_core_throughput:.2f} graphs/s", flush=True)
        print(f"  Speedup: {effective_throughput / single_core_throughput:.1f}x", flush=True)
    
    print("\nDONE", flush=True)
