"""Minimal test: isolate what operation causes torch.constant.int in backward.

Start with a tiny model fragment and progressively add complexity.
"""
import sys, os, types, contextlib, time
import torch

print(f"PyTorch: {torch.__version__}")
print("=" * 60)

# Test 1: Can we compile a simple scatter_add backward?
print("\nTest 1: Simple scatter_add + backward through compile")
print("-" * 60)

class SimpleScatterModel(torch.nn.Module):
    def __init__(self, num_features=64, output_size=2):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.output_size = output_size
    
    def forward(self, x, index):
        # x: [N, 64], index: [N]
        node_out = self.linear(x)  # [N, 1]
        # Scatter-add to aggregate
        output = torch.zeros(self.output_size, 1, device=x.device, dtype=x.dtype)
        output.index_add_(0, index, node_out)
        return output.squeeze(-1)

model1 = SimpleScatterModel().to("neuron")
model1.train()
x = torch.randn(10, 64, device="neuron")
idx = torch.tensor([0,1,0,1,0,1,0,1,0,1], device="neuron", dtype=torch.int32)

compiled1 = torch.compile(model1, backend="neuron")
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)

try:
    optimizer1.zero_grad()
    out = compiled1(x, idx)
    loss = (out**2).mean()
    loss.backward()
    optimizer1.step()
    print(f"  PASSED! loss={loss.cpu().item():.6f}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

# Test 2: scatter + einsum (simulates SO3 operations)
print("\nTest 2: scatter + einsum backward")
print("-" * 60)

class ScatterEinsumModel(torch.nn.Module):
    def __init__(self, num_features=64, num_coeffs=9, output_size=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_coeffs, num_features))
        self.output_size = output_size
    
    def forward(self, x, index):
        # x: [N, num_coeffs, num_features]
        # Einsum (simulates SO3 rotation)
        y = torch.einsum("nac, ba -> nbc", x, self.weight)
        # Scatter
        output = torch.zeros(self.output_size, y.shape[1], y.shape[2], device=x.device, dtype=x.dtype)
        output.index_add_(0, index, y)
        return output.sum()

model2 = ScatterEinsumModel().to("neuron")
model2.train()
x2 = torch.randn(10, 9, 64, device="neuron")
idx2 = torch.tensor([0,1,0,1,0,1,0,1,0,1], device="neuron", dtype=torch.int32)

compiled2 = torch.compile(model2, backend="neuron")
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

try:
    optimizer2.zero_grad()
    out2 = compiled2(x2, idx2)
    out2.backward()
    optimizer2.step()
    print(f"  PASSED! loss={out2.cpu().item():.6f}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

# Test 3: scatter + indexing (like edge_index[0], edge_index[1])
print("\nTest 3: Integer tensor indexing + scatter backward")
print("-" * 60)

class IndexSelectScatterModel(torch.nn.Module):
    def __init__(self, num_features=64, num_nodes=10, output_size=2):
        super().__init__()
        self.node_embed = torch.nn.Parameter(torch.randn(num_nodes, num_features))
        self.linear = torch.nn.Linear(num_features, 1)
        self.output_size = output_size
    
    def forward(self, edge_index, batch):
        # edge_index: [2, E], batch: [N]
        src_idx = edge_index[0]  # integer indexing!
        tgt_idx = edge_index[1]
        # Gather source embeddings
        src_embed = self.node_embed[src_idx]  # fancy indexing
        # Linear
        out = self.linear(src_embed).squeeze(-1)
        # Scatter to graph
        result = torch.zeros(self.output_size, device=out.device, dtype=out.dtype)
        result.index_add_(0, batch[tgt_idx], out)
        return result

model3 = IndexSelectScatterModel().to("neuron")
model3.train()
edge_idx = torch.tensor([[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,0]], device="neuron", dtype=torch.int32)
batch3 = torch.tensor([0,0,0,0,0,1,1,1,1,1], device="neuron", dtype=torch.int32)

compiled3 = torch.compile(model3, backend="neuron")
optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)

try:
    optimizer3.zero_grad()
    out3 = compiled3(edge_idx, batch3)
    loss3 = (out3**2).mean()
    loss3.backward()
    optimizer3.step()
    print(f"  PASSED! loss={loss3.cpu().item():.6f}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

# Test 4: PyG scatter (used in softmax)
print("\nTest 4: torch_geometric scatter backward")
print("-" * 60)

try:
    from torch_geometric.utils import scatter
    
    class PyGScatterModel(torch.nn.Module):
        def __init__(self, num_features=64, num_nodes=10):
            super().__init__()
            self.linear = torch.nn.Linear(num_features, 1)
            self.num_nodes = num_nodes
        
        def forward(self, x, index):
            out = self.linear(x).squeeze(-1)
            # scatter sum
            result = scatter(out, index, dim=0, dim_size=self.num_nodes, reduce="sum")
            return result.sum()
    
    model4 = PyGScatterModel().to("neuron")
    model4.train()
    x4 = torch.randn(20, 64, device="neuron")
    idx4 = torch.randint(0, 10, (20,), device="neuron", dtype=torch.int32)
    
    compiled4 = torch.compile(model4, backend="neuron")
    optimizer4 = torch.optim.SGD(model4.parameters(), lr=0.01)
    
    try:
        optimizer4.zero_grad()
        out4 = compiled4(x4, idx4)
        out4.backward()
        optimizer4.step()
        print(f"  PASSED! loss={out4.cpu().item():.6f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")
except ImportError as e:
    print(f"  SKIPPED: {e}")

print("\n" + "=" * 60)
print("DONE")
