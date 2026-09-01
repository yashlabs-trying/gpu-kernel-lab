"""M=1 projection kernels for decode; output-parallel and optional Split-K."""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BN': 32, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BN': 64, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
    ], key=['N', 'K'])
@triton.jit
def gemv_kernel(X, W, Y, N: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
    """One program owns BN outputs, so no atomics or reduction launch is needed."""
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accum = tl.zeros((BN,), tl.float32)
    for tile in range(0, tl.cdiv(K, BK)):
        k = tile * BK + kk
        x = tl.load(X + k, mask=k < K, other=0.)
        weights = tl.load(W + k[:, None] * N + cols[None, :],
                          mask=(k[:, None] < K) & (cols[None, :] < N), other=0.)
        accum += tl.sum(weights.to(tl.float32) * x[:, None].to(tl.float32), axis=0)
    tl.store(Y + cols, accum, mask=cols < N)


@triton.jit
def splitk_gemv_kernel(X, W, PARTIAL, N: tl.constexpr, K: tl.constexpr,
                       SPLITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Optional deep-K path: disjoint FP32 partials, deliberately no atomics."""
    pid_n, split = tl.program_id(0), tl.program_id(1)
    cols = pid_n * BN + tl.arange(0, BN)
    split_size = tl.cdiv(K, SPLITS)
    begin, end = split * split_size, tl.minimum((split + 1) * split_size, K)
    kk = tl.arange(0, BK)
    accum = tl.zeros((BN,), tl.float32)
    for tile in range(0, tl.cdiv(split_size, BK)):
        k = begin + tile * BK + kk
        valid = k < end
        x = tl.load(X + k, mask=valid, other=0.)
        w = tl.load(W + k[:, None] * N + cols[None, :],
                    mask=valid[:, None] & (cols[None, :] < N), other=0.)
        accum += tl.sum(w.to(tl.float32) * x[:, None].to(tl.float32), axis=0)
    tl.store(PARTIAL + split * N + cols, accum, mask=cols < N)


@triton.jit
def reduce_splitk_kernel(PARTIAL, Y, N: tl.constexpr, SPLITS: tl.constexpr,
                         BLOCK: tl.constexpr):
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    accum = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLITS):
        accum += tl.load(PARTIAL + split * N + cols, mask=cols < N, other=0.)
    tl.store(Y + cols, accum, mask=cols < N)


def decode_gemv(x, weight, *, split_k=1):
    """Compute `[K] @ [K,N]`; no physical padding and no implicit copies."""
    if x.ndim != 1 or weight.ndim != 2 or x.numel() != weight.shape[0]:
        raise ValueError('expected x[K] and row-major weight[K,N]')
    if not x.is_cuda or weight.device != x.device or x.dtype != weight.dtype:
        raise ValueError('same-dtype CUDA tensors required')
    if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('contiguous FP16/BF16 tensors required')
    if split_k not in (1, 2, 4, 8):
        raise ValueError('split_k must be 1, 2, 4, or 8')
    k, n = weight.shape
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    if split_k == 1:
        grid = lambda meta: (triton.cdiv(n, meta['BN']),)
        gemv_kernel[grid](x, weight, y, n, k)
    else:
        # Split-K costs a workspace and second launch; only benchmarked winners qualify.
        partial = torch.empty((split_k, n), device=x.device, dtype=torch.float32)
        splitk_gemv_kernel[(triton.cdiv(n, 64), split_k)](
            x, weight, partial, n, k, split_k, BN=64, BK=32,
            num_warps=4, num_stages=3)
        reduce_splitk_kernel[(triton.cdiv(n, 256),)](
            partial, y, n, split_k, BLOCK=256, num_warps=4)
    return y
