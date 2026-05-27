import torch

from .scatter_ops import scatter_add


def reduce_edge(inputs, edge_index, output_shape):
    # output_shape[0] is the number of nodes (output dim 0)
    # scatter_add handles creating the zero tensor internally
    return scatter_add(inputs, edge_index, output_shape[0])
