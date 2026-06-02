# When Eager Mode Outperforms NKI Kernels and torch.compile on Neuron

## Abstract

When optimizing EquiformerV3 graph neural network inference on AWS Trainium, we discovered
that no single execution strategy is optimal for all operations. Custom NKI kernels — often
considered the "ultimate" optimization — were 3.3x slower than eager PyTorch for batched
matrix multiplication with per-sample varying matrices. Similarly, `torch.compile` caused
register spills when applied to the full forward pass, making it slower than eager. The
optimal strategy is a **hybrid**: eager batched matmul, fused shared-weight matmul, and
compiled element-wise ops — yielding 43x speedup over the original implementation.

## The Model

EquiformerV3 is an equivariant graph neural network for molecular dynamics that operates
on SO(3) representations. Its attention layer performs:

1. **Merge**: combine source and target node features with edge weights
2. **Wigner rotation**: rotate features into a local frame via per-edge 25x25 matrices
3. **SO2 linear**: apply learned linear transforms per spherical-harmonic order (m)
4. **Activation**: SiLU nonlinearity
5. **Inverse rotation**: rotate back to the global frame
6. **Aggregation**: sum messages from neighbors

At production scale, this processes 9,000 edges with M=25 orders and C=128 channels per
attention layer, repeated across 8 layers.

## The Optimization Attempts

### Attempt 1: Full torch.compile

Wrapping the entire attention forward pass in `torch.compile(backend='neuron')` should,
in theory, let the compiler see the full dataflow and schedule optimally.

**Result**: 23ms compiled vs 14ms eager — **compile is 1.6x slower**.

The compiler includes the batched matmul (9000 x 25 x 25 @ 9000 x 25 x 128) in its
fusion graph. This operation dominates memory, and fusing it with surrounding ops causes
register pressure to exceed on-chip SRAM capacity, triggering DMA spills to HBM. The
spill traffic dwarfs any fusion benefit.

### Attempt 2: NKI Kernel for Wigner Rotation

The wigner rotation is `torch.bmm(wigner, x)` where `wigner` is [9000, 25, 25] and `x`
is [9000, 25, 128]. We wrote an NKI kernel (V4b) that fuses the preceding merge
operation with the rotation into a single kernel call.

**Result**: 14.9ms NKI vs 4.5ms eager bmm — **NKI is 3.3x slower**.

Why? The NKI programming model has no hardware loop instructions. Every loop is fully
unrolled at compile time. With 9,000 per-edge varying wigner matrices:
- The 25x25 tile iteration: 625 multiply-accumulate iterations
- Tiled across 128 channels: 71 output tiles
- Total: ~44,000 instructions in the generated NEFF

NKI's `nisa.nc_matmul` (Neuron Contraction Matrix Multiply) requires one operand to be
"stationary" — loaded once from SBUF and reused. But per-edge wigner matrices differ for
every edge, so there's nothing to reuse. Each edge needs its own matrix load, negating
the stationary-operand advantage.

Meanwhile, eager `torch.bmm` dispatches directly to the hardware's batched matrix
multiply unit, which is purpose-built for this pattern: many small independent matmuls
with varying operands.

### Attempt 3: Selective torch.compile

Instead of compiling everything, we compile **only element-wise subgraphs**:

| Subgraph | Eager (ms) | Compiled (ms) | Speedup |
|----------|-----------|--------------|---------|
| Merge (`a*b + c*d`) | 1.88 | 0.77 | 2.4x |
| SiLU activation | 1.71 | 0.80 | 2.1x |
| SO2 Linear | 280 (with compiler issue) | 2.58 | 109x |

For SO2 Linear, we also discovered a **fused matmul** approach (combining all m-order
weights into one block-diagonal matrix) that runs in 2.26ms — slightly faster than the
compiled 2.58ms and much simpler.

## The Hybrid Strategy

The final implementation (`enable_compile()`) selects per-op:

```
┌─────────────────┬──────────────────┬────────────────────────────────┐
│ Operation       │ Strategy         │ Why                            │
├─────────────────┼──────────────────┼────────────────────────────────┤
│ Merge (a*b+c*d) │ torch.compile    │ Fuses 4 tensors → 1 NEFF      │
│ Wigner bmm      │ Eager            │ HW batched matmul, no spills   │
│ SO2 Linear      │ Fused matmul     │ Block-diag avoids compiler bug │
│ SiLU            │ torch.compile    │ Element-wise fusion benefit    │
│ Inverse bmm     │ Eager            │ Same as forward rotation       │
│ Aggregation     │ Eager            │ Irregular indexing, not fusible │
└─────────────────┴──────────────────┴────────────────────────────────┘
```

## Results

Per attention layer (E=9000, M=25, C=128) on trn2.3xlarge:

| Strategy | Latency | Speedup |
|----------|---------|---------|
| Original eager (SO2 with subtract) | 587 ms | 1x |
| Eager + fused SO2 only | 14.98 ms | 39x |
| **Hybrid (fused SO2 + compiled elem)** | **13.71 ms** | **43x** |

Over 8 layers, the hybrid saves ~10ms total compared to fused-only.

## When Does Each Strategy Win?

### Eager wins when:
- The operation uses **hardware-specialized execution units** (batched matmul)
- Input data varies per sample (no reuse opportunity for NKI stationary model)
- The operation is memory-bandwidth efficient at its native dispatch level
- Including it in a compile graph would cause register spills

### torch.compile wins when:
- Multiple **element-wise** operations can be fused into one kernel
- Individual ops are too small to amortize dispatch overhead alone
- The subgraph fits in on-chip SRAM without spilling

### NKI wins when (not this case, but in general):
- A matrix operand is **shared** across the batch (stationary model works)
- Custom tiling can exploit data reuse patterns the compiler misses
- The iteration space is small enough that full unrolling is manageable
- You need sub-tile precision (mixed precision, custom accumulation)

### Fused matmul (manual reshape) wins when:
- Compiler bugs prevent correct/efficient code generation for the original pattern
- A simple algebraic rearrangement eliminates the problematic pattern entirely
- The resulting single large matmul maps directly to hardware matrix units

## Lessons Learned

1. **Profile before assuming compile helps.** The Neuron compiler is excellent at
   element-wise fusion but can regress on operations that already map well to hardware
   dispatch (bmm, scatter).

2. **NKI's power is data reuse, not raw speed.** If your workload has no reusable
   stationary operand, the overhead of NKI's explicit memory management exceeds the
   benefit. The hardware's native batched matmul path is highly optimized.

3. **Loop unrolling is the hidden cost of NKI.** Without hardware loop instructions,
   large iteration spaces produce enormous NEFFs. This is a fundamental architectural
   constraint, not a fixable performance issue.

4. **The optimal boundary for compile is the subgraph, not the model.** Compiling
   individual ops or entire models both underperform. The sweet spot is identifying
   fusible subgraphs that fit in SRAM and leaving hardware-optimized ops in eager.

5. **Compiler bugs sometimes have better workarounds than compile.** The SO2 Linear
   issue (NCC_ILSA902) was solved more efficiently by algebraic restructuring (fusing
   weights into a block-diagonal matrix) than by asking the compiler to fix itself.

## Applicability

This hybrid pattern applies broadly to models with mixed computational motifs:
- **Equivariant GNNs** (EquiformerV3, MACE, Allegro) — per-edge rotations + shared MLPs
- **Attention mechanisms** — softmax and scoring (compile) vs QKV matmul (eager)
- **Sparse/irregular models** — scatter/gather (eager) vs dense compute (compile)

The key insight: **hardware accelerators are not uniform execution engines.** Different
ops map to different hardware paths, and the fastest execution strategy depends on which
path the op naturally targets.
