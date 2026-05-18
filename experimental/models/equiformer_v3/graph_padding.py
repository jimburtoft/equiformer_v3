"""Graph padding utilities for EquiformerV3 torch.compile compatibility.

Provides StaticGraphData that pads molecular graphs to fixed max_nodes/max_edges,
enabling torch.compile with static shapes. The forward_static() method on
EquiformerV3_OC accepts the pre-computed tensors directly, bypassing
generate_graph() (which is decorated with @torch._dynamo.disable).

For training:
  - Pad graphs to fixed shapes for consistent tensor dimensions
  - forward_static() is fully differentiable (no @torch._dynamo.disable)
  - Energy/force losses can backpropagate through the padded forward pass
  - Use node_mask/edge_mask to exclude padding from loss computation
"""

import torch


class StaticGraphData:
    """Pre-computed graph data with static tensor shapes for torch.compile.

    All tensors are padded to max_nodes/max_edges. Padded edges have small
    nonzero distances and connect node 0->1, contributing negligible signal.
    """

    def __init__(
        self, max_nodes: int, max_edges: int, device="cpu", dtype=torch.float32
    ):
        """Initialize with static dimensions (fill with from_atoms() or from_tensors()).

        Args:
            max_nodes: Fixed number of node slots.
            max_edges: Fixed number of edge slots.
            device: Target device.
            dtype: Float dtype for distance tensors.
        """
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.device = device
        self.dtype = dtype

        # These will be populated by from_* methods
        self.atomic_numbers = None
        self.edge_index = None
        self.edge_distance = None
        self.edge_distance_vec = None
        self.batch = None
        self.batch_size = None
        self.node_mask = None
        self.edge_mask = None
        self.actual_nodes = 0
        self.actual_edges = 0

    @classmethod
    def from_positions(
        cls,
        positions,
        atomic_numbers,
        edge_index,
        batch,
        batch_size,
        max_nodes,
        max_edges,
        device="cpu",
        dtype=torch.float32,
    ):
        """Create padded graph from positions and edge connectivity.

        Computes edge_distance and edge_distance_vec from positions and edge_index.

        Args:
            positions: [N, 3] atom positions
            atomic_numbers: [N] atomic numbers (long)
            edge_index: [2, E] edge connectivity
            batch: [N] graph membership indices
            batch_size: number of graphs
            max_nodes: padding target for nodes
            max_edges: padding target for edges
            device: target device
            dtype: float dtype
        """
        actual_nodes = positions.shape[0]
        actual_edges = edge_index.shape[1]

        assert actual_nodes <= max_nodes, (
            f"Graph has {actual_nodes} nodes > max_nodes={max_nodes}"
        )
        assert actual_edges <= max_edges, (
            f"Graph has {actual_edges} edges > max_edges={max_edges}"
        )

        # Compute distances from positions
        row = edge_index[0]
        col = edge_index[1]
        distance_vec = positions[row] - positions[col]
        edge_distance = distance_vec.norm(dim=-1)

        return cls.from_tensors(
            atomic_numbers=atomic_numbers,
            edge_index=edge_index,
            edge_distance=edge_distance,
            edge_distance_vec=distance_vec,
            batch=batch,
            batch_size=batch_size,
            max_nodes=max_nodes,
            max_edges=max_edges,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_tensors(
        cls,
        atomic_numbers,
        edge_index,
        edge_distance,
        edge_distance_vec,
        batch,
        batch_size,
        max_nodes,
        max_edges,
        device="cpu",
        dtype=torch.float32,
    ):
        """Create padded graph from pre-computed tensors.

        Args:
            atomic_numbers: [N] long
            edge_index: [2, E] long
            edge_distance: [E] float (scalar distances)
            edge_distance_vec: [E, 3] float (distance vectors)
            batch: [N] long (graph membership)
            batch_size: int
            max_nodes: padding target for nodes
            max_edges: padding target for edges
            device: target device
            dtype: float dtype
        """
        actual_nodes = atomic_numbers.shape[0]
        actual_edges = edge_index.shape[1]

        assert actual_nodes <= max_nodes, (
            f"Graph has {actual_nodes} nodes > max_nodes={max_nodes}"
        )
        assert actual_edges <= max_edges, (
            f"Graph has {actual_edges} edges > max_edges={max_edges}"
        )

        obj = cls(max_nodes, max_edges, device, dtype)
        obj.actual_nodes = actual_nodes
        obj.actual_edges = actual_edges
        obj.batch_size = batch_size

        # --- Node-level tensors (padded to max_nodes) ---
        obj.atomic_numbers = _pad_long(atomic_numbers, max_nodes, device, pad_value=0)
        obj.batch = _pad_long(batch, max_nodes, device, pad_value=0)

        # --- Edge-level tensors (padded to max_edges) ---
        # Padded edges: src=0, dst=1 (ensures nonzero distance)
        edge_index_padded = torch.zeros(2, max_edges, dtype=torch.long, device=device)
        edge_index_padded[:, :actual_edges] = edge_index.to(device)
        edge_index_padded[0, actual_edges:] = 0
        edge_index_padded[1, actual_edges:] = min(1, actual_nodes - 1)
        obj.edge_index = edge_index_padded

        # Edge distances
        ed_padded = torch.ones(max_edges, dtype=dtype, device=device)
        ed_padded[:actual_edges] = edge_distance.to(device=device, dtype=dtype)
        obj.edge_distance = ed_padded

        # Edge distance vectors (padded edges get [1, 0, 0])
        edv_padded = torch.zeros(max_edges, 3, dtype=dtype, device=device)
        edv_padded[:actual_edges] = edge_distance_vec.to(device=device, dtype=dtype)
        edv_padded[actual_edges:, 0] = 1.0  # unit vector for padded edges
        obj.edge_distance_vec = edv_padded

        # Ensure no zero distances (edge_rot_mat divides by distance)
        zero_mask = obj.edge_distance < 1e-6
        if zero_mask.any():
            obj.edge_distance = obj.edge_distance + zero_mask.float() * 1.0
            obj.edge_distance_vec = obj.edge_distance_vec.clone()
            obj.edge_distance_vec[zero_mask, 0] = 1.0
            obj.edge_distance_vec[zero_mask, 1] = 0.0
            obj.edge_distance_vec[zero_mask, 2] = 0.0

        # Masks for extracting real outputs
        obj.node_mask = torch.zeros(max_nodes, dtype=torch.bool, device=device)
        obj.node_mask[:actual_nodes] = True
        obj.edge_mask = torch.zeros(max_edges, dtype=torch.bool, device=device)
        obj.edge_mask[:actual_edges] = True

        return obj

    def get_forward_static_args(self):
        """Return the argument tuple for model.forward_static()."""
        return (
            self.atomic_numbers,
            self.edge_index,
            self.edge_distance,
            self.edge_distance_vec,
            self.batch,
            self.batch_size,
        )


def _pad_long(tensor, target_size: int, device, pad_value=0):
    """Pad a long tensor's first dimension to target_size."""
    tensor = tensor.to(device)
    actual = tensor.shape[0]
    if actual >= target_size:
        return tensor[:target_size]
    pad_shape = (target_size - actual,) + tensor.shape[1:]
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=device)
    return torch.cat([tensor, padding], dim=0)


def create_random_graph(
    num_atoms: int,
    max_neighbors: int,
    max_radius: float,
    batch_size: int = 1,
    device="cpu",
    dtype=torch.float32,
):
    """Create a random molecular graph for testing.

    Args:
        num_atoms: atoms per molecule
        max_neighbors: max neighbors per atom
        max_radius: max distance for edges
        batch_size: number of molecules
        device: target device
        dtype: float dtype

    Returns:
        dict with atomic_numbers, edge_index, edge_distance, edge_distance_vec, batch, batch_size
    """
    total_atoms = num_atoms * batch_size

    # Random positions in a box
    positions = torch.randn(total_atoms, 3, device=device, dtype=dtype) * 3.0

    # Random atomic numbers (H=1 to Xe=54)
    atomic_numbers = torch.randint(1, 55, (total_atoms,), device=device)

    # Batch indices
    batch = torch.arange(batch_size, device=device).repeat_interleave(num_atoms)

    # Build edges: for each atom, connect to nearest neighbors within radius
    edge_src = []
    edge_dst = []
    for b in range(batch_size):
        start = b * num_atoms
        end = start + num_atoms
        pos_b = positions[start:end]
        # Pairwise distances
        diff = pos_b.unsqueeze(0) - pos_b.unsqueeze(1)  # [N, N, 3]
        dist = diff.norm(dim=-1)  # [N, N]
        # Mask self-loops and beyond radius
        dist.fill_diagonal_(float("inf"))
        dist[dist > max_radius] = float("inf")
        # Keep top-k neighbors per atom
        k = min(max_neighbors, num_atoms - 1)
        _, topk_idx = dist.topk(k, dim=1, largest=False)
        for i in range(num_atoms):
            for j_idx in range(k):
                j = topk_idx[i, j_idx].item()
                if dist[i, j] < float("inf"):
                    edge_src.append(start + i)
                    edge_dst.append(start + j)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    # Compute distances
    row = edge_index[0]
    col = edge_index[1]
    edge_distance_vec = positions[row] - positions[col]
    edge_distance = edge_distance_vec.norm(dim=-1)

    return {
        "atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "edge_distance": edge_distance,
        "edge_distance_vec": edge_distance_vec,
        "batch": batch,
        "batch_size": batch_size,
        "positions": positions,
        "node_mask": torch.ones(total_atoms, dtype=torch.bool, device=device),
    }
