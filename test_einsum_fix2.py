"""Test: isolate einsum issue and find matmul replacement."""

import torch
import time

print(f"PyTorch: {torch.__version__}")
print("=" * 60)

# The actual SO3 einsum in equiformer_v3:
# torch.einsum('nac, ba -> nbc', embedding, to_m)
# embedding: [N, A, C] where A = (lmax+1)^2 = 9, C = num_channels = 64
# to_m: [B, A] where B = num_m_coefficients = 9
# output: [N, B, C]
# output[n,b,c] = sum_a embedding[n,a,c] * to_m[b,a]

N, A, C, B = 100, 9, 64, 9

x_cpu = torch.randn(N, A, C)
w_cpu = torch.randn(B, A)

# einsum version
out_einsum = torch.einsum("nac, ba -> nbc", x_cpu, w_cpu)
print(f"  einsum output shape: {out_einsum.shape}")  # [N, B, C]

# matmul version: output[n,b,c] = sum_a w[b,a] * x[n,a,c]
# w.unsqueeze(0) = [1, B, A], x = [N, A, C] -> matmul([1,B,A], [N,A,C]) -> [N, B, C]
out_matmul = torch.matmul(w_cpu.unsqueeze(0), x_cpu)  # [1,B,A] @ [N,A,C] -> [N,B,C]
print(f"  matmul output shape: {out_matmul.shape}")
print(f"  Equivalence: max_diff = {(out_einsum - out_matmul).abs().max().item():.2e}")

# Also test: 'nac, ab -> nbc'
# output[n,b,c] = sum_a x[n,a,c] * M[a,b]
# [N,C,A] @ [A,B] -> [N,C,B] -> permute -> [N,B,C]
w2_cpu = torch.randn(A, B)
out_einsum2 = torch.einsum("nac, ab -> nbc", x_cpu, w2_cpu)
out_matmul2 = torch.matmul(x_cpu.permute(0, 2, 1), w2_cpu).permute(0, 2, 1)
print(
    f"\n  einsum 'nac, ab -> nbc' equivalence: max_diff = {(out_einsum2 - out_matmul2).abs().max().item():.2e}"
)

# Now test on Neuron
print("\n" + "=" * 60)
print("Neuron compile tests")
print("=" * 60)

# Test A: matmul version compiles for training
print("\nTest A: matmul version compile training")
print("-" * 60)


class MatmulSO3(torch.nn.Module):
    def __init__(self, A=9, B=9, C=64):
        super().__init__()
        self.to_m = torch.nn.Parameter(torch.randn(B, A))

    def forward(self, x):
        # x: [N, A, C]
        # einsum('nac, ba -> nbc', x, to_m) via matmul
        return torch.matmul(self.to_m.unsqueeze(0), x)  # [1,B,A] @ [N,A,C] -> [N,B,C]


model_a = MatmulSO3().to("neuron")
model_a.train()
x_n = torch.randn(100, 9, 64, device="neuron")
compiled_a = torch.compile(model_a, backend="neuron")
opt_a = torch.optim.SGD(model_a.parameters(), lr=0.01)

try:
    for step in range(3):
        t0 = time.time()
        opt_a.zero_grad()
        out = compiled_a(x_n)
        loss = (out**2).mean()
        loss.backward()
        opt_a.step()
        print(
            f"  Step {step + 1}: loss={loss.cpu().item():.6f} time={time.time() - t0:.1f}s"
        )
    print("  MATMUL SO3 COMPILE: PASSED!")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

# Test B: einsum version
print("\nTest B: einsum version compile training")
print("-" * 60)


class EinsumSO3(torch.nn.Module):
    def __init__(self, A=9, B=9, C=64):
        super().__init__()
        self.to_m = torch.nn.Parameter(torch.randn(B, A))

    def forward(self, x):
        return torch.einsum("nac, ba -> nbc", x, self.to_m)


model_b = EinsumSO3().to("neuron")
model_b.train()
x_n2 = torch.randn(100, 9, 64, device="neuron")
compiled_b = torch.compile(model_b, backend="neuron")
opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)

try:
    for step in range(3):
        t0 = time.time()
        opt_b.zero_grad()
        out = compiled_b(x_n2)
        loss = (out**2).mean()
        loss.backward()
        opt_b.step()
        print(
            f"  Step {step + 1}: loss={loss.cpu().item():.6f} time={time.time() - t0:.1f}s"
        )
    print("  EINSUM SO3 COMPILE: PASSED!")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

# Test C: Full GNN-like pattern with matmul
print("\nTest C: Full GNN pattern (indexing + matmul + scatter)")
print("-" * 60)


class FullGNNPattern(torch.nn.Module):
    def __init__(self, num_nodes=50, A=9, C=64, output_size=2):
        super().__init__()
        self.node_embed = torch.nn.Parameter(torch.randn(num_nodes, A, C))
        self.rotation = torch.nn.Parameter(torch.randn(A, A))
        self.linear = torch.nn.Linear(C, 1)
        self.output_size = output_size

    def forward(self, edge_index, batch):
        # Gather
        src = self.node_embed[edge_index[0]]  # [E, A, C]
        # Rotate (matmul instead of einsum)
        rotated = torch.matmul(
            self.rotation.unsqueeze(0), src
        )  # [1,A,A] @ [E,A,C] -> [E,A,C]
        # Pool over coefficients
        pooled = rotated.mean(dim=1)  # [E, C]
        # Project
        scalar = self.linear(pooled).squeeze(-1)  # [E]
        # Scatter
        output = torch.zeros(self.output_size, device=scalar.device, dtype=scalar.dtype)
        output.index_add_(0, batch[edge_index[1]], scalar)
        return output


model_c = FullGNNPattern().to("neuron")
model_c.train()
# Create edge_index and batch
E = 200
edge_idx = torch.stack([torch.randint(0, 50, (E,)), torch.randint(0, 50, (E,))]).to(
    device="neuron", dtype=torch.int32
)
batch_c = torch.cat(
    [torch.zeros(25, dtype=torch.int32), torch.ones(25, dtype=torch.int32)]
).to("neuron")

compiled_c = torch.compile(model_c, backend="neuron")
opt_c = torch.optim.SGD(model_c.parameters(), lr=0.01)

try:
    for step in range(3):
        t0 = time.time()
        opt_c.zero_grad()
        out = compiled_c(edge_idx, batch_c)
        loss = (out**2).mean()
        loss.backward()
        opt_c.step()
        print(
            f"  Step {step + 1}: loss={loss.cpu().item():.6f} time={time.time() - t0:.1f}s"
        )
    print("  FULL GNN PATTERN COMPILE: PASSED!")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

print("\n" + "=" * 60)
print("DONE")
