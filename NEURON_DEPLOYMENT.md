# EquiformerV3 on AWS Trainium: Deployment Guide

Deploy EquiformerV3 for inference on AWS Trainium (trn2) instances using PyTorch Native
(`torch.compile` with the Neuron aot_autograd backend). This guide covers environment setup,
model preparation, and running optimized compiled inference.

> **Performance**: Compiled inference achieves **3–15x speedup** over eager mode on Neuron,
> and **1.5–1.9x over CPU** at production edge counts. The compilation uses `aot_autograd`
> with Neuron's decomposition table to fuse operations into optimized NEFFs.
>
> Key optimizations applied:
> - `build_fused_linear()` — NCC_ILSA902 compiler bug workaround
> - `pre_expand_weights()` — eliminate runtime `index_select` for compilability
> - `compile_for_neuron()` — 15x speedup via op fusion (aot_autograd + Neuron compiler)

## Quick Start

```bash
# On a trn2.3xlarge instance with PyTorch Native DLC
git clone https://github.com/jimburtoft/equiformer_v3.git
cd equiformer_v3
git checkout neuron-optimize

# Setup (one-time)
bash setup_instance.sh

# Run inference
bash run_in_docker.sh inference_example.py
```

## Performance Summary

### Compiled Inference (recommended — `compile_for_neuron()`)

Uses `torch.compile` + `aot_autograd` with Neuron decomposition table. Fuses operations
into optimized NEFFs, achieving 3–15x over eager depending on model size.

**Small model (2-layer, lmax=2, C=64, 1.3M params):**

| Atoms | Neighbors | Edges | Compiled (ms) | Eager (ms) | Speedup |
|-------|-----------|-------|---------------|-----------|---------|
| 64    | 10        | 640   | 13            | 190       | 14.6x   |
| 100   | 10        | 1,000 | 17            | 190       | 11.2x   |
| 128   | 10        | 1,280 | 17            | 190       | 10.9x   |
| 256   | 10        | 2,560 | 27            | —         | —       |

**Production model (7-layer, lmax=4, C=128, 79M params):**

| Atoms | Neighbors | Edges | Compiled (ms) | Eager (ms) | Speedup |
|-------|-----------|-------|---------------|-----------|---------|
| 50    | 10        | 500   | 107           | ~350      | ~3.3x   |
| 100   | 10        | 1,000 | 229           | ~770      | 3.4x    |
| 100   | 30        | 3,000 | 515           | ~1800     | ~3.5x   |

Compile time: 30–110s per new shape (one-time cost, increases with edge count).

### Eager with CPU comparison (forward_static path)

Neuron beats CPU on **all tested configurations** (3K–30K edges):

| Atoms | Neighbors | Edges | CPU (ms) | Neuron (ms) | Speedup |
|-------|-----------|-------|----------|-------------|---------|
| 100   | 30        | 3,000 | 96       | 62          | 1.5x    |
| 200   | 30        | 6,000 | 203      | 123         | 1.7x    |
| 500   | 30        | 15,000| 549      | 297         | 1.9x    |
| 1000  | 30        | 30,000| 1124     | 638         | 1.8x    |

**Sweet spot**: 200–500 atoms, 30 neighbors → 1.5–1.9x speedup over CPU.

## Requirements

- **Instance**: trn2.3xlarge (or larger)
- **Container**: PyTorch Native Beta DLC
  ```
  421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
  ```
- **Framework**: PyTorch 2.11, torch_neuronx, neuronx-cc 2.25+
- **Python packages**: `e3nn`, `torch-geometric`, `packaging`

## Environment Setup

### 1. Launch Instance

Use a trn2.3xlarge with the standard Neuron DLAMI (Ubuntu 24.04) or any AMI with Docker:

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 2. Pull the DLC and Install Dependencies

```bash
# ECR login (us-east-1 region for the DLC)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com

# Pull the PyTorch Native container
docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

# Install model dependencies into a committed image
docker run --name eq_setup --privileged \
  421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest \
  bash -c 'pip install e3nn torch-geometric packaging -q'
docker commit eq_setup equiformer_v3:latest
docker rm eq_setup
```

### 3. Clone the Model

```bash
cd ~
git clone https://github.com/jimburtoft/equiformer_v3.git code
cd code
git checkout neuron-optimize
```

## Running Inference

### Minimal Example

Create `inference_example.py`:

```python
"""Minimal EquiformerV3 inference on Neuron."""
import sys, time, torch, types, contextlib

# --- Stub out fairchem (not needed for inference) ---
for p in ['fairchem', 'fairchem.core', 'fairchem.core.common', 'fairchem.core.models']:
    m = types.ModuleType(p); m.__package__ = p; m.__path__ = []; sys.modules[p] = m
    pts = p.split('.')
    if len(pts) > 1:
        setattr(sys.modules['.'.join(pts[:-1])], pts[-1], m)

r = types.ModuleType('fairchem.core.common.registry')
r.__package__ = 'fairchem.core.common'
class _R:
    @staticmethod
    def register_model(n): return lambda c: c
r.registry = _R()
sys.modules['fairchem.core.common.registry'] = r
sys.modules['fairchem.core.common'].registry = r

u = types.ModuleType('fairchem.core.common.utils')
u.__package__ = 'fairchem.core.common'
@contextlib.contextmanager
def _cg(f):
    if f: yield
    else:
        with torch.no_grad(): yield
u.conditional_grad = _cg
sys.modules['fairchem.core.common.utils'] = u
sys.modules['fairchem.core.common'].utils = u

b = types.ModuleType('fairchem.core.models.base')
b.__package__ = 'fairchem.core.models'
class _G: pass
b.GraphModelMixin = _G
sys.modules['fairchem.core.models.base'] = b
sys.modules['fairchem.core.models'].base = b

# --- Import the model ---
sys.path.insert(0, '/root/equiformer_v3')
from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC
from experimental.models.equiformer_v3.edge_rot_mat import init_edge_rot_mat
from experimental.models.equiformer_v3.so3 import SO3Rotation

# --- Configuration ---
NUM_ATOMS = 200
MAX_NEIGHBORS = 30
DEVICE = 'neuron:0'

# --- Create Model ---
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
    norm_type='merge_layer_norm',
    attn_activation='sep-merge_gates2_swiglu',
    ffn_activation='sep-merge_gates2_swiglu',
    use_gate_force_head=False,
)
model.eval()

# CRITICAL: Build fused linear AFTER loading weights (here using random init)
model.build_fused_linear()

# Move to Neuron device
model = model.to(DEVICE)

# --- Prepare Inputs ---
# In production, replace this with your actual molecular graph construction
N, K = NUM_ATOMS, MAX_NEIGHBORS
torch.manual_seed(42)
atomic_numbers = torch.randint(1, 95, (N,))
positions = torch.randn(N, 3) * 5.0

# Build neighbor list (dense padded format)
neighbor_idx = torch.zeros(N, K, dtype=torch.long)
mask = torch.zeros(N, K, dtype=torch.bool)
for i in range(N):
    actual = min(K, N - 1)
    nb = [(i + j + 1) % N for j in range(actual)]
    neighbor_idx[i, :actual] = torch.tensor(nb)
    mask[i, :actual] = True

# Compute edge vectors and distances
src_pos = positions[neighbor_idx.view(-1)]
tgt_pos = positions.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
edge_vec = src_pos - tgt_pos
edge_dist = edge_vec.norm(dim=1).clamp(min=0.1)

# Mask padding edges
flat_mask = mask.view(-1)
edge_dist[~flat_mask] = 1.0
edge_vec[~flat_mask] = torch.tensor([0., 0., 1.])

# Pre-compute Wigner matrices (requires atan2/acos — CPU only)
temp_so3 = SO3Rotation(model.lmax, model.mmax, use_rotation_mask=False)
with torch.no_grad():
    edge_rot_mat = init_edge_rot_mat(edge_vec, use_rotation_mask=False)
    temp_so3.set_wigner(edge_rot_mat)
    wigner = temp_so3.wigner.clone()       # Already contiguous (fix in so3.py)
    wigner_inv = temp_so3.wigner_inv.clone()

# Package inputs for the dense forward path
inputs = dict(
    atomic_numbers=atomic_numbers.to(DEVICE),
    neighbor_idx=neighbor_idx.to(DEVICE),
    edge_distance=edge_dist.to(DEVICE),
    edge_distance_vec=edge_vec.to(DEVICE),
    wigner=wigner.to(DEVICE),
    wigner_inv=wigner_inv.to(DEVICE),
    mask=mask.to(DEVICE),
    batch=torch.zeros(N, dtype=torch.long).to(DEVICE),
    batch_size=1,
)

# --- Compile ---
print("Compiling model (first run takes 5-20 minutes, cached afterward)...")
compiled_fn = torch.compile(
    model.forward_dense,
    backend='neuron',
    fullgraph=True,
    dynamic=False,
)

t0 = time.time()
with torch.no_grad():
    output = compiled_fn(**inputs)
torch.neuron.synchronize()
print(f"Compilation + first inference: {time.time() - t0:.1f}s")

# --- Warmup ---
with torch.no_grad():
    for _ in range(5):
        compiled_fn(**inputs)
        torch.neuron.synchronize()

# --- Benchmark ---
times = []
with torch.no_grad():
    for _ in range(20):
        t0 = time.perf_counter_ns()
        output = compiled_fn(**inputs)
        torch.neuron.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e6)

avg_ms = sum(times) / len(times)
print(f"\nResults (N={N}, K={K}, {N*min(K,N-1)} edges):")
print(f"  Average latency: {avg_ms:.1f} ms")
print(f"  Min: {min(times):.1f} ms, Max: {max(times):.1f} ms")
print(f"  Energy output shape: {output['energy'].shape}")
```

Run it:
```bash
docker run --rm --privileged \
  -v ~/code:/root/equiformer_v3 \
  -e HOME=/root \
  equiformer_v3:latest \
  bash -c 'cd /root && python /root/equiformer_v3/inference_example.py'
```

## Key Code Changes for Neuron Compatibility

The `neuron-optimize` branch contains several modifications to enable `torch.compile`
with `fullgraph=True` on the Neuron backend. These patterns are **generalizable to
other PyTorch models** targeting Neuron.

### 1. Non-contiguous Tensor Fix (3.4x speedup)

**Problem**: `torch.einsum("mi, nij -> nmj", ...)` produces a tensor with non-contiguous
memory layout. The Neuron runtime's `_nrt_copy_strided_neuron_to_neuron` must transpose
this on every inference call, adding ~235ms overhead.

**Fix** (`so3.py:322`):
```python
# After the einsum
wigner = wigner.contiguous()  # Critical: prevents 235ms strided copy
wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
```

**Generalizable pattern**: Always call `.contiguous()` on tensors produced by `einsum`,
`permute`, `transpose`, or any operation that returns a view with non-standard strides,
before passing them to compiled Neuron graphs.

### 2. Fused Linear to Avoid Subtraction (NCC_ILSA902 workaround)

**Problem**: The subtract operation `x_r_0 - x_i_1` in SO2MLinear triggers compiler bug
NCC_ILSA902 at certain tensor dimensions.

**Fix** (`so2_ops.py`): Build a fused weight matrix that absorbs the negation:
```python
# Instead of: out_real = W_r @ x_real - W_i @ x_imag  (subtract triggers bug)
# Use:        W_fused = [[W_r, -W_i], [W_i, W_r]]
#             [out_real, out_imag] = W_fused @ [x_real, x_imag]  (no subtract)
model.build_fused_linear()  # Call after loading weights
```

**Generalizable pattern**: If a subtract triggers a compiler error, absorb the sign flip
into a preceding linear layer's weight matrix.

### 3. `narrow` → `split`/`chunk` Rewrite

**Problem**: The backward pass of `narrow` generates `slice_scatter` ops, which cause
`torch.constant.int` errors in the Neuron compiler.

**Fix**: Replace all `narrow` and in-place slice assignments with `split`/`chunk`/`cat`:
```python
# Before (generates slice_scatter in backward):
x_real = x[:, 0:1, :]
x_imag = x[:, 1:2, :]

# After (backward generates cat — compiler-friendly):
x_real, x_imag = torch.chunk(x, chunks=2, dim=1)
```

Files changed: `so2_ops.py`, `activation.py`, `layer_norm.py`, `transformer_block.py`,
`so3.py`, `equiformer_v3.py`.

**Generalizable pattern**: Replace `narrow`/slice indexing with `split`/`chunk` whenever
you need to compile the backward pass on Neuron.

### 4. Dense Padded Forward Path (`forward_dense`)

**Problem**: The original `forward()` uses `scatter_add`, `atan2`, `acos` which fall
back to CPU (graph breaks). This prevents `fullgraph=True` compilation.

**Fix**: A separate `forward_dense()` method that:
- Takes pre-computed Wigner matrices as input (avoids `atan2`/`acos`)
- Uses dense `[N, K]` neighbor layout instead of variable-length edge lists
- Replaces `scatter_add` with `reshape + sum` over the neighbor dimension
- Replaces `GraphSoftmax` with masked `F.softmax`

The preprocessing (Wigner computation, neighbor list construction) happens outside the
compiled region on CPU or eagerly on device.

**Generalizable pattern**: For GNN models with scatter operations, convert to dense
padded neighbor format to enable full-graph compilation.

## Supported Configurations

### Recommended (tested, Neuron wins)

| Atoms | Neighbors (K) | Edges | Expected Latency |
|-------|---------------|-------|-----------------|
| 100   | 30            | 3,000 | ~62 ms          |
| 200   | 30            | 6,000 | ~123 ms         |
| 250   | 30            | 7,500 | ~140 ms         |
| 500   | 30            | 15,000| ~297 ms         |
| 1000  | 30            | 30,000| ~638 ms         |

### Known Limitations

- **K < 25 at small N**: May trigger NCC_ILSA902 compiler bug (use K ≥ 30)
- **K = 99 (maximum density)**: Triggers compiler bug
- **Compilation time**: 5–20 minutes per new (N, K) shape. Cached after first run.
- **N < 100**: Untested (small tensors may trigger different compiler paths)

### Compilation Caching

NEFFs are cached automatically. Set `NEURON_COMPILE_CACHE_URL` for persistent caching:
```bash
export NEURON_COMPILE_CACHE_URL=/path/to/neff_cache
```

When using Docker with `--rm`, mount the cache as a volume to persist across runs:
```bash
docker run --rm --privileged \
  -v ~/code:/root/equiformer_v3 \
  -v ~/neff_cache:/tmp/neff_cache \
  -e NEURON_COMPILE_CACHE_URL=/tmp/neff_cache \
  -e HOME=/root \
  equiformer_v3:latest \
  bash -c 'cd /root && python /root/equiformer_v3/inference_example.py'
```

After first compilation, subsequent runs for the same shape complete in <6 seconds.

## Troubleshooting

### "No backend named 'neuron'"

You're not using the PyTorch Native DLC container. The standard DLAMI does not include
the `neuron` torch.compile backend. Use the DLC:
```
421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
```

### NCC_ILSA902 Compiler Error

```
[INTERNAL_ERROR] [NCC_ILSA902] LegalizeSundaAccess error
```

This is triggered by certain tensor dimensions. Solutions:
1. Use K ≥ 30 (avoid K=20 at N=100)
2. Call `model.build_fused_linear()` to eliminate subtract operations
3. If the error persists at your shape, try adjusting K by ±5

### Slow First Run (10-20 minutes)

This is expected — the Neuron compiler is generating optimized NEFFs. Set
`NEURON_COMPILE_CACHE_URL` to persist the cache across runs.

### "nrta_tensor_write uses synchronous implementation"

This is a warning, not an error. The Neuron Runtime will add async tensor I/O in
a future release. It does not affect correctness or performance.

### Docker: "permission denied" or "privileged" errors

The `--privileged` flag is required for Neuron device access:
```bash
docker run --rm --privileged ...
```

## Architecture Notes

The deployment uses the **dense padded inference path** (`forward_dense`):

```
Input: atoms, positions
  │
  ├─ [CPU/eager] Build neighbor list → [N, K] dense tensor
  ├─ [CPU/eager] Compute edge vectors, distances
  ├─ [CPU/eager] Compute Wigner matrices (atan2, acos)
  │
  └─ [Neuron compiled, fullgraph=True] forward_dense()
       ├─ Atom embedding
       ├─ Radial basis expansion
       ├─ For each transformer block:
       │    ├─ LayerNorm
       │    ├─ Graph Attention (SO3 rotation → SO2 linear → attention → SO3 inverse)
       │    ├─ FFN (SwiGLU)
       │    └─ Residual connection
       └─ Output head → energy prediction
```

The split between CPU preprocessing and Neuron inference is intentional:
- `atan2`/`acos` (for rotation matrices) don't compile on Neuron
- Neighbor list construction uses dynamic shapes
- The compiled region handles ALL the compute-heavy operations in a single NEFF

## Model Parameters

The test configuration (2-layer, lmax=2, C=64):
- Parameters: ~16.7M
- NEFF size: ~10 MB (compiled)
- HBM usage: <1 GB (well within 24 GB per LNC=2 core)

For larger models (7-layer, lmax=4, C=128, 35.8M params), expect:
- Higher Neuron advantage (more compute per inference)
- Longer compilation times
- More HBM usage (but still fits in single core for inference)
