"""Graph padding utilities for making EquiformerV2 compatible with torch.compile.

The fundamental issue: EquiformerV2's forward pass has two dynamic dimensions:
  - num_nodes: number of atoms in the batch (varies per molecule)
  - num_edges: number of neighbor pairs (varies per structure)

These propagate to every tensor in the model. torch.compile needs static shapes.

Solution: Pre-compute the graph with padding to fixed max_nodes/max_edges.
The model's generate_graph (which has dynamic ops like nonzero filtering) is
bypassed entirely by providing a pre-computed GraphData object.
"""

import torch
from fairchem.core.models.base import GraphData


class StaticGraphData:
    """Pre-computed graph data with static tensor shapes for torch.compile.

    Bypass generate_graph() entirely by providing pre-computed distances/edges.
    All tensors padded to max_nodes/max_edges. Padded edges have small nonzero
    distances and point to node 0, making their contribution negligible.
    """

    def __init__(self, data, max_nodes: int, max_edges: int, device="cpu"):
        """Pre-compute and pad graph data to static dimensions.

        Args:
            data: Original graph data with .pos, .atomic_numbers, .edge_index, etc.
                  Edge distances are computed from pos and edge_index.
            max_nodes: Static max number of nodes.
            max_edges: Static max number of edges.
            device: Target device.
        """
        actual_nodes = data.pos.shape[0]
        actual_edges = data.edge_index.shape[1]

        assert actual_nodes <= max_nodes, (
            f"Graph has {actual_nodes} nodes > max_nodes={max_nodes}"
        )
        assert actual_edges <= max_edges, (
            f"Graph has {actual_edges} edges > max_edges={max_edges}"
        )

        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.actual_nodes = actual_nodes
        self.actual_edges = actual_edges

        # --- Node-level tensors (padded to max_nodes) ---
        self.pos = _pad(data.pos, max_nodes, device)
        self.atomic_numbers = _pad(
            data.atomic_numbers.long(), max_nodes, device, pad_value=0
        )
        self.batch = _pad(data.batch, max_nodes, device, pad_value=0)
        self.tags = _pad(
            getattr(data, "tags", torch.ones(actual_nodes, dtype=torch.long)),
            max_nodes,
            device,
            pad_value=0,
        )
        self.natoms = data.natoms.to(device)
        self.cell = data.cell.to(device) if hasattr(data, "cell") else None

        # --- Edge-level tensors (padded to max_edges) ---
        # Pad edge_index: padded edges go 0->1 (nonzero distance guaranteed)
        edge_index_padded = torch.zeros(2, max_edges, dtype=torch.long, device=device)
        edge_index_padded[:, :actual_edges] = data.edge_index.to(device)
        # Padded edges: src=0, dst=1 (ensures nonzero distance since pos[0]!=pos[1] generically)
        edge_index_padded[0, actual_edges:] = 0
        edge_index_padded[1, actual_edges:] = min(1, actual_nodes - 1)
        self.edge_index = edge_index_padded

        # Compute edge distances from positions
        pos_dev = self.pos
        row = self.edge_index[0]
        col = self.edge_index[1]
        distance_vec = pos_dev[row] - pos_dev[col]

        # Apply PBC offsets for real edges
        if hasattr(data, "cell_offsets") and data.cell is not None:
            cell_offsets_padded = _pad(data.cell_offsets, max_edges, device)
            cell = data.cell.float().to(device)
            # cell is [batch, 3, 3], repeat for all edges
            cell_expanded = cell.repeat(max_edges, 1, 1)  # [max_edges, 3, 3]
            offsets = (
                cell_offsets_padded.float().unsqueeze(1).bmm(cell_expanded).squeeze(1)
            )  # [max_edges, 3]
            distance_vec = distance_vec + offsets

        edge_distance = distance_vec.norm(dim=-1)

        # Ensure no zero distances or zero vectors for padded edges.
        # edge_rot_mat divides by distance, so zeros produce NaN.
        # Replace zero-distance edges with a unit vector [1, 0, 0].
        zero_mask = edge_distance < 1e-6
        if zero_mask.any():
            edge_distance = edge_distance + zero_mask.float() * 1.0
            # Set distance_vec for zero edges to [1, 0, 0] * distance
            distance_vec = distance_vec.clone()
            distance_vec[zero_mask, 0] = 1.0
            distance_vec[zero_mask, 1] = 0.0
            distance_vec[zero_mask, 2] = 0.0

        self.edge_distance = edge_distance  # [max_edges]
        self.edge_distance_vec = distance_vec  # [max_edges, 3]

        # Neighbors (for compatibility, not used when bypassing generate_graph)
        self.neighbors = torch.tensor([max_edges], device=device)

        # Node/edge masks for extracting real outputs
        self.node_mask = torch.zeros(max_nodes, dtype=torch.bool, device=device)
        self.node_mask[:actual_nodes] = True
        self.edge_mask = torch.zeros(max_edges, dtype=torch.bool, device=device)
        self.edge_mask[:actual_edges] = True

    def to_graph_data(self):
        """Convert to GraphData for direct use (bypassing generate_graph)."""
        return GraphData(
            edge_index=self.edge_index,
            edge_distance=self.edge_distance,
            edge_distance_vec=self.edge_distance_vec,
            cell_offsets=torch.zeros_like(self.edge_distance_vec),
            offset_distances=torch.zeros_like(self.edge_distance),
            neighbors=self.neighbors,
            node_offset=0,
            batch_full=self.batch,
            atomic_numbers_full=self.atomic_numbers,
        )


def _pad(tensor, target_size: int, device, pad_value=0):
    """Pad tensor's first dimension to target_size."""
    tensor = tensor.to(device)
    actual = tensor.shape[0]
    if actual >= target_size:
        return tensor[:target_size]
    pad_shape = (target_size - actual,) + tensor.shape[1:]
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=device)
    return torch.cat([tensor, padding], dim=0)


def unpad_output(output_embedding, node_mask):
    """Extract real node embeddings from padded output.

    Args:
        output_embedding: SO3_Embedding with .embedding shape [max_nodes, coeffs, channels]
        node_mask: Boolean mask [max_nodes], True for real nodes.

    Returns:
        Tensor of shape [actual_nodes, coeffs, channels]
    """
    return output_embedding.embedding[node_mask]
