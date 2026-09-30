"""Llama RMSNorm candidates specialized for contiguous inference tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(
    x_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    x = tl.load(x_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_size + epsilon)
    tl.store(output_ptr + row * hidden_size + offsets, x * inverse_rms * weight, mask=mask)


@triton.jit
def _fused_add_rmsnorm(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_output_ptr,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    x = tl.load(x_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * hidden_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    combined = x + residual
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(combined * combined, axis=0) / hidden_size + epsilon)
    tl.store(residual_output_ptr + row * hidden_size + offsets, combined, mask=mask)
    tl.store(
        output_ptr + row * hidden_size + offsets,
        combined * inverse_rms * weight,
        mask=mask,
    )


def _validate(*tensors: torch.Tensor) -> tuple[int, int, int]:
    x, weight = tensors[:2]
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("CUDA tensors are required")
    if x.shape[-1] != weight.numel() or weight.ndim != 1:
        raise ValueError("weight must match the hidden dimension")
    if x.dtype != weight.dtype or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("input and weight must be contiguous with matching dtypes")
    for tensor in tensors[2:]:
        if tensor.shape != x.shape or tensor.dtype != x.dtype or not tensor.is_contiguous():
            raise ValueError("all activation tensors must match the contiguous input")
    hidden_size = x.shape[-1]
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65_536:
        raise ValueError("hidden dimension is too large for the single-stage candidate")
    return x.numel() // hidden_size, hidden_size, block_size


def rmsnorm_into(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    *,
    num_warps: int = 8,
) -> None:
    rows, hidden_size, block_size = _validate(x, weight, output)
    _rmsnorm[(rows,)](
        x,
        weight,
        output,
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )


def fused_add_rmsnorm_into(
    output: torch.Tensor,
    residual_output: torch.Tensor,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-5,
    *,
    num_warps: int = 8,
) -> None:
    rows, hidden_size, block_size = _validate(
        x, weight, residual, output, residual_output
    )
    _fused_add_rmsnorm[(rows,)](
        x,
        residual,
        weight,
        output,
        residual_output,
        hidden_size=hidden_size,
        epsilon=epsilon,
        block_size=block_size,
        num_warps=num_warps,
    )
