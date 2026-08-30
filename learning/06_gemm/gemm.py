"""Tiled Triton GEMM: C[M,N] = A[M,K] @ B[K,N]."""

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    m_size: tl.constexpr, n_size: tl.constexpr, k_size: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """One program accumulates one BLOCK_M x BLOCK_N output tile."""
    program_id = tl.program_id(axis=0)
    programs_m = tl.cdiv(m_size, BLOCK_M)
    programs_n = tl.cdiv(n_size, BLOCK_N)

    # Group nearby M tiles so they reuse B tiles in cache more effectively.
    programs_per_group = GROUP_SIZE_M * programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * GROUP_SIZE_M
    group_m = tl.minimum(programs_m - first_program_m, GROUP_SIZE_M)
    program_m = first_program_m + (program_id % group_m)
    program_n = (program_id % programs_per_group) // group_m

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    # These pointer grids describe vectorized, coalesced tiles.
    a_tile_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_tile_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    # The whole output tile remains FP32 and on-chip throughout the K loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(k_size, BLOCK_K)):
        k_mask = k_start * BLOCK_K + offsets_k < k_size
        a = tl.load(a_tile_ptrs, mask=(offsets_m[:, None] < m_size) & k_mask[None, :], other=0.0)
        b = tl.load(b_tile_ptrs, mask=k_mask[:, None] & (offsets_n[None, :] < n_size), other=0.0)
        accumulator += tl.dot(a, b)
        a_tile_ptrs += BLOCK_K * stride_ak
        b_tile_ptrs += BLOCK_K * stride_bk

    # Global output is written exactly once, after every K tile is reduced.
    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    c_mask = (offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def pytorch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def triton_gemm(
    a: torch.Tensor, b: torch.Tensor, *,
    block_m: int = 64, block_n: int = 64, block_k: int = 32,
    num_warps: int = 4, num_stages: int = 3, group_size_m: int = 8,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if not a.is_cuda or not b.is_cuda or a.dtype != b.dtype:
        raise ValueError("A and B must be same-dtype CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("this first GEMM requires contiguous row-major inputs")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("this Tensor Core lesson supports FP16 and BF16")

    m_size, k_size = a.shape
    n_size = b.shape[1]
    c = torch.empty((m_size, n_size), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m_size, block_m) * triton.cdiv(n_size, block_n),)
    gemm_kernel[grid](
        a, b, c, m_size, n_size, k_size,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_SIZE_M=group_size_m, num_warps=num_warps, num_stages=num_stages,
    )
    return c

