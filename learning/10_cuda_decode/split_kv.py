"""Single-query GQA split-KV candidate; no padding tokens, dropout, or window mask.

All supplied cache positions must be valid past/current positions. This is
sequence partitioning on ONE GPU, not multi-GPU tensor parallelism.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def partial(Q,K,V,O,L,VALID,S:tl.constexpr,D:tl.constexpr,GQA:tl.constexpr,
            KH:tl.constexpr,KS:tl.constexpr,VH:tl.constexpr,VS:tl.constexpr,
            SPLITS:tl.constexpr,TILE:tl.constexpr,HAS_VALID:tl.constexpr):
    head, split = tl.program_id(0), tl.program_id(1)
    t = split*TILE+tl.arange(0,TILE)
    valid_length=tl.load(VALID)+1 if HAS_VALID else S
    d = tl.arange(0,D)
    q = tl.load(Q+head*D+d).to(tl.float32)
    keys = tl.load(K+(head//GQA)*KH+t[:,None]*KS+d[None,:],t[:,None]<valid_length,other=0).to(tl.float32)
    scores = tl.sum(keys*q[None,:],1)*(D**-0.5)
    scores = tl.where(t<valid_length,scores,-float('inf'))
    maximum = tl.max(scores,0)
    active=split*TILE<valid_length
    safe_maximum=tl.where(active,maximum,0.0)
    probability = tl.exp(scores-safe_maximum)
    denominator = tl.sum(probability,0)
    values = tl.load(V+(head//GQA)*VH+t[:,None]*VS+d[None,:],t[:,None]<valid_length,other=0).to(tl.float32)
    out = tl.where(active,tl.sum(probability[:,None]*values,0)/denominator,0.0)
    tl.store(O+(head*SPLITS+split)*D+d,out)
    tl.store(L+head*SPLITS+split,tl.where(active,safe_maximum+tl.log(denominator),-float('inf')))


@triton.jit
def merge(O,L,Y,D:tl.constexpr,SPLITS:tl.constexpr,B:tl.constexpr):
    head = tl.program_id(0)
    s,d = tl.arange(0,B),tl.arange(0,D)
    l = tl.load(L+head*SPLITS+s,s<SPLITS,other=-float('inf'))
    p = tl.exp(l-tl.max(l,0))
    out = tl.load(O+(head*SPLITS+s[:,None])*D+d[None,:],s[:,None]<SPLITS,other=0)
    result = tl.sum(p[:,None]*out,0)/tl.sum(p,0)
    tl.store(Y+head*D+d,result)


def split_kv_attention(q,k,v,tile=128,*,valid_position=None,output=None,local=None,lse=None):
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
    if valid_position is not None:
        if valid_position.shape!=(1,) or valid_position.dtype not in (torch.int32,torch.int64) or valid_position.device!=q.device or not valid_position.is_contiguous():
            raise ValueError('valid_position must be a colocated contiguous integer scalar tensor')
    dummy=q
    output = torch.empty_like(q) if output is None else output
    local = torch.empty((h,splits,d),device=q.device,dtype=torch.float32) if local is None else local
    lse = torch.empty((h,splits),device=q.device,dtype=torch.float32) if lse is None else lse
    if output.shape!=q.shape or output.dtype!=q.dtype or output.device!=q.device or not output.is_contiguous():
        raise ValueError('output workspace mismatch')
    if local.shape!=(h,splits,d) or local.dtype!=torch.float32 or local.device!=q.device or not local.is_contiguous():
        raise ValueError('local workspace mismatch')
    if lse.shape!=(h,splits) or lse.dtype!=torch.float32 or lse.device!=q.device or not lse.is_contiguous():
        raise ValueError('lse workspace mismatch')
    with torch.cuda.device(q.device):
        partial[(h,splits)](q,k,v,local,lse,valid_position if valid_position is not None else dummy,
            s,d,h//k.shape[0],k.stride(0),k.stride(1),v.stride(0),v.stride(1),splits,tile,valid_position is not None,num_warps=4)
        merge[(h,)](local,lse,output,d,splits,triton.next_power_of_2(splits),num_warps=4)
    return output
