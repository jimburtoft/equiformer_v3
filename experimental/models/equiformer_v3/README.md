# EquiformerV3 Neuron Inference Optimization

Hybrid optimization for EquiformerV3 attention layers on AWS Trainium (trn2), achieving
**43x speedup** per attention layer at production scale (9000 edges).

## Why These Changes Were Made

The EquiformerV3 attention layer has three distinct computational patterns, each with a
different optimal execution strategy on Neuron hardware:

1. **SO2 Linear (shared-weight matmul)**: The original implementation uses per-m subtraction
   to apply separate linear transforms per spherical-harmonic order. This triggers a Neuron
   compiler issue (NCC_ILSA902) causing catastrophic slowdown (~587ms/layer). The fix fuses
   all m-order weights into a single block-diagonal matrix, reducing to one matmul (2.26ms).

2. **Element-wise ops (merge, activation)**: Simple `a*b + c*d` and SiLU operations are
   memory-bound with many small tensors. `torch.compile(backend='neuron')` fuses these into
   single NEFF calls, yielding 2.4x speedup (0.77ms vs 1.88ms for merge).

3. **Batched matrix multiply (wigner rotation)**: Per-edge varying wigner matrices mean each
   of 9000 edges has its own 25x25 rotation. Eager `torch.bmm` uses optimized hardware batched
   matmul directly. Both `torch.compile` (spills when bmm included) and NKI (no hardware loops,
   44K unrolled instructions) are slower.

The hybrid `enable_compile()` method selects the best strategy per op automatically.

## Repository

```
https://github.com/jimburtoft/equiformer_v3
Branch: neuron-optimize
```

## Instance Requirements

- **Instance type**: trn2.3xlarge (sa-east-1 or any region with trn2)
- **AMI**: Deep Learning AMI Neuron (Ubuntu 24.04) 20260522 (SDK 2.30)
- **Docker**: Required — the PyTorch Native DLC container provides torch 2.11
- **DLC image**: `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest`

## Install

SSH into a trn2.3xlarge instance, then:

```bash
# 1. Clone the repository
cd /home/ubuntu
git clone https://github.com/jimburtoft/equiformer_v3.git code
cd code
git checkout neuron-optimize

# 2. Run the setup script (pulls DLC, installs dependencies, commits image)
bash setup_instance.sh
```

This pulls the PyTorch Native Docker container, installs `e3nn` and `torch-geometric`
inside it, and commits the result as `equiformer_v3:latest`.

## Run Scripts Inside Docker

All scripts must run inside the Docker container (host Python does not have torch 2.11):

```bash
# Generic usage
bash run_in_docker.sh <script.py> [args...]

# The script path is relative to the repo root (/home/ubuntu/code/)
```

The container mounts `/home/ubuntu/code` and sets `NEURON_RT_VISIBLE_CORES=0` (single device).

## Test: Per-Layer Benchmark

Benchmarks three strategies on a single attention layer (E=9000, M=25, C=128):

```bash
bash run_in_docker.sh experimental/models/equiformer_v3/bench_compile.py
```

Expected output:
```
============================================================
EquiformerV3 Attention Layer Benchmark
  N=300, K=30, E=9000, M=25, C=128
============================================================

[1/3] Eager (original SO2, with subtract issue)...
  EAGER (original): avg=587.xx ms, min=5xx.xx ms

[2/3] Eager + fused_linear (NCC_ILSA902 workaround)...
  EAGER + fused SO2: avg=14.98 ms, min=14.xx ms

[3/3] Hybrid: fused SO2 + compiled element-wise...
  HYBRID (fused SO2 + compiled elem): avg=13.71 ms, min=13.xx ms

============================================================
SUMMARY
  Eager (original SO2):     587.xx ms
  Eager + fused SO2:        14.98 ms  (39.2x vs baseline)
  Hybrid (best of both):    13.71 ms  (42.8x vs baseline, 1.09x vs fused)

  Hybrid saves: 1.27 ms per layer vs fused-only
  Over 8 layers: 10.2 ms total savings
============================================================
```

## Benchmark: Full Model (8 layers)

To benchmark the full EquiformerV3 model (8 TransBlockV3 layers, lmax=4, C=128):

```bash
bash run_in_docker.sh benchmark_large_model.py
```

This runs the production-scale model with `enable_compile()` applied to all layers and
reports end-to-end latency for a single forward pass.

## API Usage

### In your own code

```python
from experimental.models.equiformer_v3.transformer_block import TransBlockV3

# After model construction:
for block in model.blocks:
    block.enable_compile()  # Enables hybrid: fused SO2 + compiled element-wise

# Then run forward as normal (first call triggers compilation)
with torch.no_grad():
    output = model(input_data)
```

### Method hierarchy

| Method | What it does | When to use |
|--------|-------------|-------------|
| `enable_compile()` | Full hybrid: fused SO2 + compiled element-wise | Production (best perf) |
| `build_fused_linear()` | Only fuses SO2 (no compile) | If compile is unavailable |

`enable_compile()` supersedes `build_fused_linear()` — no need to call both.

## File Guide

| File | Purpose |
|------|---------|
| `transformer_block.py` | Main: `enable_compile()`, `forward_dense` with hybrid branching |
| `so2_ops.py` | `build_fused_linear()` and `forward_fused()` for SO2Linear |
| `bench_compile.py` | Per-layer benchmark (eager vs fused vs hybrid) |
| `bench_full_layer.py` | Per-op timing breakdown |
| `bench_ops_v2.py` | Individual operation benchmarks |
