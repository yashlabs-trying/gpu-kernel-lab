"""Single-query GQA split-KV candidate; no padding tokens, dropout, or window mask.

All supplied cache positions must be valid past/current positions. This is
sequence partitioning on ONE GPU, not multi-GPU tensor parallelism.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def partial(Q,K,V,O,L,S:tl.constexpr,D:tl.constexpr,GQA:tl.constexpr,
            KH:tl.constexpr,KS:tl.constexpr,VH:tl.constexpr,VS:tl.constexpr,
            SPLITS:tl.constexpr,TILE:tl.constexpr):
    head, split = tl.program_id(0), tl.program_id(1)
    t = split*TILE+tl.arange(0,TILE)
    d = tl.arange(0,D)
    q = tl.load(Q+head*D+d).to(tl.float32)
    keys = tl.load(K+(head//GQA)*KH+t[:,None]*KS+d[None,:],t[:,None]<S,other=0).to(tl.float32)
    scores = tl.sum(keys*q[None,:],1)*(D**-0.5)
    scores = tl.where(t<S,scores,-float('inf'))
    maximum = tl.max(scores,0)
    probability = tl.exp(scores-maximum)
    denominator = tl.sum(probability,0)
    values = tl.load(V+(head//GQA)*VH+t[:,None]*VS+d[None,:],t[:,None]<S,other=0).to(tl.float32)
    out = tl.sum(probability[:,None]*values,0)/denominator
    tl.store(O+(head*SPLITS+split)*D+d,out)
    tl.store(L+head*SPLITS+split,maximum+tl.log(denominator))


@triton.jit
def merge(O,L,Y,D:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
    head = tl.program_id(0)
    s,d = tl.arange(0,B),tl.arange(0,D)
    l = tl.load(L+head*SPLITS+s,s<SPLITS,other=-float('inf'))
    p = tl.exp(l-tl.max(l,0))
    out = tl.load(O+(head*SPLITS+s[:,None])*D+d[None,:],s[:,None]<SPLITS,other=0)
    result = tl.sum(p[:,None]*out,0)/tl.sum(p,0)
    tl.store(Y+head*D+d,result)


def split_kv_attention(q,k,v,tile=128):
    if q.ndim != 2 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError('q[Hq,D], k/v[Hkv,S,D] required')
    h,d = q.shape
    if h == 0 or k.shape[0] == 0 or k.shape[1] == 0 or d not in (64,128) or k.shape[2] != d or h % k.shape[0]:
        raise ValueError('nonempty GQA-compatible heads with D=64/128 required')
    if not q.is_cuda or q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError('same-device FP16/BF16 CUDA inputs required')
    if not q.is_contiguous() or k.stride(-1) != 1 or v.stride(-1) != 1 or tile not in (64,128,256):
        raise ValueError('contiguous q, unit last cache stride, tile 64/128/256 required')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v)):
        raise ValueError('inference only')
    s = k.shape[1]
    splits = triton.cdiv(s,tile)
    if splits > 256:
        raise ValueError('experimental merge limited to 256 splits')
    output = torch.empty_like(q)
    local = torch.empty((h,splits,d),device=q.device,dtype=torch.float32)
    lse = torch.empty((h,splits),device=q.device,dtype=torch.float32)
    with torch.cuda.device(q.device):
        partial[(h,splits)](q,k,v,local,lse,s,d,h//k.shape[0],k.stride(0),k.stride(1),v.stride(0),v.stride(1),splits,tile,num_warps=4)
        merge[(h,)](local,lse,output,d,splits,triton.next_power_of_2(splits),num_warps=4)
    return output
