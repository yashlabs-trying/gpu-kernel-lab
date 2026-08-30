"""Standalone and fused residual RMSNorm in PyTorch and Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr, hidden_size, epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program loads, reduces, normalizes, and stores one complete row."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size
    row_start = row * hidden_size

    # Contiguous columns give coalesced access. FP32 protects the reduction.
    x = tl.load(x_ptr + row_start + columns, mask=mask, other=0.0).to(tl.float32)
    squared_sum = tl.sum(x * x, axis=0)
    reciprocal_rms = tl.rsqrt(squared_sum / hidden_size + epsilon)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    output = x * reciprocal_rms * weight
    tl.store(output_ptr + row_start + columns, output, mask=mask)


@triton.jit
def residual_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, output_ptr, residual_output_ptr,
    hidden_size, epsilon: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Fuse residual addition, residual save, and RMSNorm into one program."""
    row = tl.program_id(axis=0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < hidden_size
    row_start = row * hidden_size

    x = tl.load(x_ptr + row_start + columns, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row_start + columns, mask=mask, other=0.0
    ).to(tl.float32)
    combined = x + residual
    squared_sum = tl.sum(combined * combined, axis=0)
    reciprocal_rms = tl.rsqrt(squared_sum / hidden_size + epsilon)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)

    tl.store(residual_output_ptr + row_start + columns, combined, mask=mask)
    tl.store(
        output_ptr + row_start + columns,
        combined * reciprocal_rms * weight,
        mask=mask,
    )


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(variance + epsilon)
    return (normalized * weight.float()).to(x.dtype)


def pytorch_residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_fp32 = x.float() + residual.float()
    variance = combined_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = combined_fp32 * torch.rsqrt(variance + epsilon)
    normalized = (normalized * weight.float()).to(x.dtype)
    return normalized, combined_fp32.to(x.dtype)


def _validate(x: torch.Tensor, weight: torch.Tensor, block_size: int) -> tuple[int, int]:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if x.ndim < 1 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        raise ValueError("weight length must equal the input's last dimension")
    if x.dtype != weight.dtype:
        raise ValueError("input and weight dtypes must match")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("this kernel requires contiguous tensors")
    hidden_size = x.shape[-1]
    if block_size < hidden_size or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two covering the hidden dimension")
    if block_size > 65_536:
        raise ValueError("use a multi-stage RMSNorm for block sizes above 65,536")
    return x.numel() // hidden_size, hidden_size


def triton_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6,
    block_size: int | None = None, num_warps: int = 4,
) -> torch.Tensor:
    hidden_size = x.shape[-1]
    block_size = block_size or triton.next_power_of_2(hidden_size)
    rows, hidden_size = _validate(x, weight, block_size)
    output = torch.empty_like(x)
    rmsnorm_kernel[(rows,)](
        x, weight, output, hidden_size, epsilon=epsilon,
        BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return output


def triton_residual_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    epsilon: float = 1e-6, block_size: int | None = None, num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.shape != x.shape or residual.dtype != x.dtype or not residual.is_contiguous():
        raise ValueError("residual must match the contiguous input")
    hidden_size = x.shape[-1]
    block_size = block_size or triton.next_power_of_2(hidden_size)
    rows, hidden_size = _validate(x, weight, block_size)
    output = torch.empty_like(x)
    residual_output = torch.empty_like(x)
    residual_rmsnorm_kernel[(rows,)](
        x, residual, weight, output, residual_output, hidden_size,
        epsilon=epsilon, BLOCK_SIZE=block_size, num_warps=num_warps,
    )
    return output, residual_output
