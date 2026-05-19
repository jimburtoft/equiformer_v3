"""Custom scatter operations with explicit backward for torch.compile compatibility.

The standard torch.index_add_ generates a backward graph containing
torch.constant.int operations that fail to lower through torch-mlir -> StableHLO.
By wrapping in a custom autograd.Function with explicit backward, we avoid
the problematic auto-generated backward graph.

Usage:
    Replace:
        output.index_add_(0, index, src)
    With:
        output = scatter_add(src, index, output_size, dim=0)
"""

import torch
from torch.autograd import Function


class ScatterAdd(Function):
    """Differentiable scatter-add: output[index[i]] += src[i] along dim 0.

    Forward: equivalent to torch.zeros(output_size, ...).index_add_(0, index, src)
    Backward w.r.t. src: grad_src = grad_output[index]  (gather by index)
    """

    @staticmethod
    def forward(ctx, src, index, output_size):
        """
        Args:
            src: [N, ...] tensor of values to scatter
            index: [N] int32/int64 index tensor (values in [0, output_size))
            output_size: int, size of output dimension 0
        Returns:
            output: [output_size, ...] tensor with scattered values
        """
        ctx.save_for_backward(index)
        ctx.src_shape = src.shape

        output = torch.zeros(
            output_size,
            *src.shape[1:],
            device=src.device,
            dtype=src.dtype,
        )
        output.index_add_(0, index, src)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: gather grad_output at index positions."""
        (index,) = ctx.saved_tensors
        # grad_src[i] = grad_output[index[i]]
        grad_src = grad_output[index]
        return grad_src, None, None


class ScatterAddInplace(Function):
    """Differentiable scatter-add that adds to an existing output tensor.

    Forward: output += scatter(src, index)
    This handles the case where output already has values (e.g., initialized to zeros
    but could also be non-zero in some formulations).
    """

    @staticmethod
    def forward(ctx, output, src, index):
        """
        Args:
            output: [M, ...] tensor to scatter into (will be cloned)
            src: [N, ...] tensor of values to scatter
            index: [N] int32/int64 index tensor
        Returns:
            result: [M, ...] tensor with scattered values added
        """
        ctx.save_for_backward(index)
        ctx.src_shape = src.shape
        ctx.output_shape = output.shape

        result = output.clone()
        result.index_add_(0, index, src)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: grad passes through to output, gather for src."""
        (index,) = ctx.saved_tensors
        # grad w.r.t. output: identity (grad passes through)
        grad_output_input = grad_output
        # grad w.r.t. src: gather
        grad_src = grad_output[index]
        return grad_output_input, grad_src, None


def scatter_add(src, index, output_size):
    """Scatter-add src into a new zero tensor of size [output_size, ...].

    Drop-in replacement for:
        output = torch.zeros(output_size, *src.shape[1:], ...)
        output.index_add_(0, index, src)

    Args:
        src: [N, ...] source tensor
        index: [N] integer index tensor
        output_size: int, first dimension of output

    Returns:
        [output_size, ...] tensor
    """
    return ScatterAdd.apply(src, index, output_size)


def scatter_add_inplace(output, src, index):
    """Scatter-add src into existing output tensor.

    Drop-in replacement for:
        output.index_add_(0, index, src)

    Note: Returns a new tensor (not truly in-place) for autograd compatibility.

    Args:
        output: [M, ...] existing tensor
        src: [N, ...] source tensor
        index: [N] integer index tensor

    Returns:
        [M, ...] tensor with src scattered in
    """
    return ScatterAddInplace.apply(output, src, index)
