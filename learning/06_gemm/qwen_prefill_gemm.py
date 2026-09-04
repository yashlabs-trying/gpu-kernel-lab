"""Shape-bucketed BF16/FP16 GEMM candidate for Qwen prefill.

This is an experimental baseline. It must beat torch.mm/cuBLAS at exact model
shapes before integration. Decode M=1 intentionally remains out of scope.
"""
import torch
import triton
import triton.language as tl


PREFILL_CONFIGS = [
    triton.Config({'BM': 32, 'BN': 64, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=PREFILL_CONFIGS, key=['M_BUCKET', 'N', 'K'])
@triton.jit
def qwen_prefill_gemm_kernel(A, B, C, M, M_BUCKET: tl.constexpr,
                             N: tl.constexpr, K: tl.constexpr,
                             LINEAR_WEIGHT: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    count_m, count_n = tl.cdiv(M_BUCKET, BM), tl.cdiv(N, BN)
    group_width = GROUP_M * count_n
    group = pid // group_width
    first_m = group * GROUP_M
    group_m = tl.minimum(count_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_m)
    pid_n = (pid % group_width) // group_m
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    for start in range(0, tl.cdiv(K, BK)):
        k = start * BK + kk
        a = tl.load(A + rows[:, None] * K + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0.)
        if LINEAR_WEIGHT:
            b = tl.load(B + cols[None, :] * K + k[:, None],
                        mask=(k[:, None] < K) & (cols[None, :] < N), other=0.)
        else:
            b = tl.load(B + k[:, None] * N + cols[None, :],
                        mask=(k[:, None] < K) & (cols[None, :] < N), other=0.)
        accum += tl.dot(a, b)
    tl.store(C + rows[:, None] * N + cols[None, :], accum,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def _bucket_m(m):
    """Limit binary/autotune keys while masking, not physically padding, rows."""
    for bucket in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        if m <= bucket:
            return bucket
    raise ValueError('prefill candidate supports M <= 4096')


def qwen_prefill_gemm(a, b):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError('expected A[M,K] and B[K,N]')
    if not a.is_cuda or b.device != a.device or a.dtype != b.dtype:
        raise ValueError('same-dtype tensors on one CUDA device required')
    if a.dtype not in (torch.float16, torch.bfloat16) or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError('contiguous FP16/BF16 tensors required')
    m, k = a.shape
    n = b.shape[1]
    if not m or not n or not k:
        return a @ b
    bucket = _bucket_m(m)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(bucket, meta['BM']) * triton.cdiv(n, meta['BN']),)
    qwen_prefill_gemm_kernel[grid](a, b, out, m, bucket, n, k, False)
    return out


def qwen_prefill_linear(x, weight):
    """`F.linear(x, weight)` for contiguous bias-free `[N,K]` weights."""
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError('expected X[...,K] and weight [N,K]')
    if not x.is_cuda or weight.device != x.device or x.dtype != weight.dtype:
        raise ValueError('same-dtype tensors on one CUDA device required')
    if x.dtype not in (torch.float16,torch.bfloat16) or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError('contiguous FP16/BF16 tensors required')
    k=x.shape[-1]; m=x.numel()//k; n=weight.shape[0]; bucket=_bucket_m(m)
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    grid=lambda meta:(triton.cdiv(bucket,meta['BM'])*triton.cdiv(n,meta['BN']),)
    qwen_prefill_gemm_kernel[grid](x.view(m,k),weight,out,m,bucket,n,k,True)
    return out.view(*x.shape[:-1],n)
