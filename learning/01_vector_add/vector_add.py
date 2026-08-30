"""PyTorch reference and Triton implementation of output[i] = x[i] + y[i]."""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Each Triton program processes BLOCK_SIZE consecutive elements."""
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Adjacent lanes load adjacent addresses, producing coalesced accesses.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def pytorch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Trusted result used to check our kernel."""
    return x + y


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Validate the inputs, allocate output, and launch the Triton kernel."""
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("inputs must have matching shapes and dtypes")
    if x.ndim != 1 or not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("this first kernel requires contiguous 1D vectors")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    output = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block_size),)
    vector_add_kernel[grid](
        x, y, output, x.numel(), BLOCK_SIZE=block_size, num_warps=4
    )
    return output

