"""Inference-only RMSNorm preserving Qwen's pre-weight dtype rounding.

Matching operation boundaries is not a promise of bitwise equality: parallel
reduction order and rsqrt implementations can differ from PyTorch.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def qwen_rmsnorm_kernel(X, W, Y, N: tl.constexpr, EPS: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < N
    # Neighboring logical lanes address neighboring elements of one row.
    x = tl.load(X + row * N + col, valid, other=0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    normalized = x * tl.rsqrt(variance + EPS)
    # Important: materialize Qwen's low-precision rounding BEFORE weight multiply.
    rounded = normalized.to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + col, valid, other=0).to(tl.float32)
    tl.store(Y + row * N + col, rounded * weight, valid)


@triton.jit
def qwen_residual_rmsnorm_kernel(X, RESIDUAL, W, Y, RESIDUAL_OUT,
                                 N: tl.constexpr, EPS: tl.constexpr,
                                 BLOCK: tl.constexpr):
    """Fuse Qwen's low-precision residual add with the following RMSNorm."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < N
    offsets = row * N + col
    x = tl.load(X + offsets, valid, other=0).to(tl.float32)
    residual = tl.load(RESIDUAL + offsets, valid, other=0).to(tl.float32)
    # A standalone torch add writes the residual stream in the input dtype.
    combined = (x + residual).to(X.dtype.element_ty)
    tl.store(RESIDUAL_OUT + offsets, combined, valid)
    combined_fp32 = combined.to(tl.float32)
    variance = tl.sum(combined_fp32 * combined_fp32, axis=0) / N
    normalized = combined_fp32 * tl.rsqrt(variance + EPS)
    rounded = normalized.to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + col, valid, other=0).to(tl.float32)
    tl.store(Y + offsets, rounded * weight, valid)


def qwen_rmsnorm_reference(x, weight, epsilon=1e-6):
    """Qwen eager ordering; tests also compare the installed model class."""
    fp32 = x.float()
    normalized = fp32 * torch.rsqrt(fp32.pow(2).mean(-1, keepdim=True) + epsilon)
    return weight * normalized.to(x.dtype)


def qwen_residual_rmsnorm_reference(x, residual, weight, epsilon=1e-6):
    combined = x + residual
    return qwen_rmsnorm_reference(combined, weight, epsilon), combined


def triton_qwen_rmsnorm(x, weight, epsilon=1e-6, num_warps=4):
    """Contiguous, matching-dtype CUDA tensors only; no silent copies/fallbacks."""
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError('input must have a nonempty last dimension')
    if not x.is_cuda or weight.device != x.device:
        raise ValueError('input and weight must be on the same CUDA device')
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise ValueError('input and weight must have matching FP32/FP16/BF16 dtype')
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError('weight must be a vector matching the last dimension')
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('contiguous input and weight required')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    if num_warps not in (2, 4, 8):
        raise ValueError('num_warps must be 2, 4, or 8')
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError('inference only; use torch.inference_mode()')
    n = x.shape[-1]
    block = triton.next_power_of_2(n)
    if block > 4096:
        raise ValueError('this model-targeted implementation supports widths <= 4096')
    output = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            qwen_rmsnorm_kernel[(x.numel() // n,)](
                x, weight, output, n, epsilon, block,
                num_warps=num_warps, enable_fp_fusion=False,
            )
    return output


def triton_qwen_residual_rmsnorm(x, residual, weight, epsilon=1e-6, num_warps=4):
    """Return `(normalized, residual_stream)` without materializing add then norm."""
    if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
        raise ValueError('residual must match input shape, dtype, and device')
    if not residual.is_contiguous():
        raise ValueError('contiguous residual required')
    # Validate without launching the standalone kernel.
    if x.ndim < 1 or x.shape[-1] == 0 or not x.is_cuda or weight.device != x.device:
        raise ValueError('valid CUDA input and colocated weight required')
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise ValueError('input and weight must have matching FP32/FP16/BF16 dtype')
    if weight.ndim != 1 or weight.numel() != x.shape[-1] or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('contiguous input and matching vector weight required')
    if not math.isfinite(epsilon) or epsilon <= 0 or num_warps not in (2, 4, 8):
        raise ValueError('invalid epsilon or num_warps')
    if torch.is_grad_enabled() and (x.requires_grad or residual.requires_grad or weight.requires_grad):
        raise ValueError('inference only; use torch.inference_mode()')
    n = x.shape[-1]
    block = triton.next_power_of_2(n)
    if block > 4096:
        raise ValueError('this model-targeted implementation supports widths <= 4096')
    output, residual_output = torch.empty_like(x), torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            qwen_residual_rmsnorm_kernel[(x.numel() // n,)](
                x, residual, weight, output, residual_output, n, epsilon, block,
                num_warps=num_warps, enable_fp_fusion=False,
            )
    return output, residual_output
