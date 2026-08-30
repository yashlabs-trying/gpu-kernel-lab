"""Numerically stable, one-load row-wise Triton softmax."""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, output_ptr, columns, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < columns
    row_start = row * columns

    # Load once and retain FP32 values through both reductions.
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    tl.store(output_ptr + row_start + offsets, output, mask=mask)


def pytorch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def triton_softmax(
    x: torch.Tensor, block_size: int | None = None, num_warps: int = 4
) -> torch.Tensor:
    if not x.is_cuda or x.ndim < 1 or not x.is_contiguous():
        raise ValueError("input must be a contiguous CUDA tensor with at least one dimension")
    columns = x.shape[-1]
    block_size = block_size or triton.next_power_of_2(columns)
    if block_size < columns or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two covering the row")
    if block_size > 65_536:
        raise ValueError("large rows require a multi-stage or tiled softmax")
    rows = x.numel() // columns
    output = torch.empty_like(x)
    softmax_kernel[(rows,)](
        x, output, columns, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    return output

