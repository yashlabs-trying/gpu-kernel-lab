"""Gate projection + up projection + model-ordered SwiGLU in one Triton kernel."""
import torch
import triton
import triton.language as tl


CONFIGS = [
    # Two FP32 accumulator tiles are live. Keep tiles deliberately smaller than
    # a single-output GEMM and let measurements choose among these candidates.
    triton.Config({'BM': 16, 'BN': 32, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BM': 32, 'BN': 32, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BM': 32, 'BN': 64, 'BK': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 32, 'BK': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=CONFIGS, key=['M_BUCKET', 'N', 'K'])
@triton.jit
def projection_swiglu_kernel(X, WG, WU, Y, M, M_BUCKET: tl.constexpr,
                             N: tl.constexpr, K: tl.constexpr,
                             MODEL_ORDERED: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr,
                             BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    count_m, count_n = tl.cdiv(M_BUCKET, BM), tl.cdiv(N, BN)
    group_width = GROUP_M * count_n
    group = pid // group_width
    first_m = group * GROUP_M
    group_m = tl.minimum(count_m - first_m, GROUP_M)
    pid_m = first_m + pid % group_m
    pid_n = (pid % group_width) // group_m
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for tile in range(0, tl.cdiv(K, BK)):
        k = tile * BK + kk
        # X is loaded once and reused by both tensor-core dot products.
        x = tl.load(X + rows[:, None] * K + k[None, :],
                    mask=(rows[:, None] < M) & (k[None, :] < K), other=0.)
        # nn.Linear weights are stored [N,K]; K is the contiguous direction.
        wg = tl.load(WG + cols[:, None] * K + k[None, :],
                     mask=(cols[:, None] < N) & (k[None, :] < K), other=0.)
        wu = tl.load(WU + cols[:, None] * K + k[None, :],
                     mask=(cols[:, None] < N) & (k[None, :] < K), other=0.)
        gate += tl.dot(x, tl.trans(wg))
        up += tl.dot(x, tl.trans(wu))
    if MODEL_ORDERED:
        # Preserve the eager model's storage-dtype boundaries without global
        # gate/up/activation tensors. These casts remain inside the program.
        gate = gate.to(X.dtype.element_ty).to(tl.float32)
        up = up.to(X.dtype.element_ty).to(tl.float32)
        activated = (gate * tl.sigmoid(gate)).to(X.dtype.element_ty).to(tl.float32)
        result = activated * up
    else:
        result = gate * tl.sigmoid(gate) * up
    tl.store(Y + rows[:, None] * N + cols[None, :], result,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.autotune(configs=[
    triton.Config({'BN': 16, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BN': 32, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BN': 64, 'BK': 32}, num_warps=8, num_stages=3),
], key=['N', 'K'])
@triton.jit
def projection_swiglu_gemv_kernel(X, WG, WU, Y, N: tl.constexpr,
                                  K: tl.constexpr, MODEL_ORDERED: tl.constexpr,
                                  BN: tl.constexpr, BK: tl.constexpr):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BN,), tl.float32)
    up = tl.zeros((BN,), tl.float32)
    for tile in range(0, tl.cdiv(K, BK)):
        k = tile * BK + kk
        x = tl.load(X + k, mask=k < K, other=0.).to(tl.float32)
        wg = tl.load(WG + cols[:, None] * K + k[None, :],
                     mask=(cols[:, None] < N) & (k[None, :] < K), other=0.).to(tl.float32)
        wu = tl.load(WU + cols[:, None] * K + k[None, :],
                     mask=(cols[:, None] < N) & (k[None, :] < K), other=0.).to(tl.float32)
        gate += tl.sum(wg * x[None, :], axis=1)
        up += tl.sum(wu * x[None, :], axis=1)
    if MODEL_ORDERED:
        gate = gate.to(X.dtype.element_ty).to(tl.float32)
        up = up.to(X.dtype.element_ty).to(tl.float32)
        activated = (gate * tl.sigmoid(gate)).to(X.dtype.element_ty).to(tl.float32)
        result = activated * up
    else:
        result = gate * tl.sigmoid(gate) * up
    tl.store(Y + cols, result, mask=cols < N)


def _validate(x, gate_weight, up_weight):
    if x.ndim < 1 or gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError('expected X[...,K] and matching weights [N,K]')
    if x.shape[-1] != gate_weight.shape[1]:
        raise ValueError('input K must match weight K')
    tensors=(x,gate_weight,up_weight)
    if not all(t.is_cuda and t.device == x.device and t.dtype == x.dtype and t.is_contiguous() for t in tensors):
        raise ValueError('contiguous same-dtype tensors on one CUDA device required')
    if x.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError('FP16/BF16 only')


def fused_projection_swiglu(x, gate_weight, up_weight, *, model_ordered=True):
    """Dispatch M=1 to GEMV and M>1 to a tensor-core tiled GEMM."""
    _validate(x,gate_weight,up_weight)
    n,k=gate_weight.shape
    original_shape=x.shape[:-1]
    m=x.numel()//k
    if m == 1:
        y=torch.empty(n,device=x.device,dtype=x.dtype)
        grid=lambda meta:(triton.cdiv(n,meta['BN']),)
        projection_swiglu_gemv_kernel[grid](x.reshape(-1),gate_weight,up_weight,y,n,k,model_ordered)
        return y.view(*original_shape,n)
    x=x.view(m,k)
    for bucket in (16,32,64,128,256,512,1024,2048,4096):
        if m <= bucket: break
    else: raise ValueError('M must be <= 4096')
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    grid=lambda meta:(triton.cdiv(bucket,meta['BM'])*triton.cdiv(n,meta['BN']),)
    projection_swiglu_kernel[grid](x,gate_weight,up_weight,y,m,bucket,n,k,model_ordered)
    return y.view(*original_shape,n)
