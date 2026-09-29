"""GQA decode attention that reads physical paged K/V through block tables."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q,K,V,TABLE,LENGTHS,O,LSE,LAYER,
             H:tl.constexpr,HKV:tl.constexpr,D:tl.constexpr,
             BLOCK:tl.constexpr,TABLE_WIDTH:tl.constexpr,SPLITS:tl.constexpr,
             TILE:tl.constexpr,
             KSL:tl.constexpr,KSB:tl.constexpr,KSH:tl.constexpr,KST:tl.constexpr,
             VSL:tl.constexpr,VSB:tl.constexpr,VSH:tl.constexpr,VST:tl.constexpr):
    bh,split=tl.program_id(0),tl.program_id(1)
    batch,head=bh//H,bh%H
    length=tl.load(LENGTHS+batch)
    token=split*TILE+tl.arange(0,TILE)
    active=token<length
    logical_block=token//BLOCK
    physical=tl.load(TABLE+batch*TABLE_WIDTH+logical_block,mask=active,other=0)
    offset=token%BLOCK
    dimension=tl.arange(0,D)
    q=tl.load(Q+bh*D+dimension).to(tl.float32)
    kv_head=head//(H//HKV)
    key=tl.load(K+LAYER*KSL+physical[:,None]*KSB+kv_head*KSH+
                offset[:,None]*KST+dimension[None,:],mask=active[:,None],other=0).to(tl.float32)
    score=tl.sum(key*q[None,:],axis=1)*(D**-0.5)
    score=tl.where(active,score,-float('inf'))
    maximum=tl.max(score,axis=0)
    split_active=split*TILE<length
    safe_max=tl.where(split_active,maximum,0.0)
    probability=tl.exp(score-safe_max)
    denominator=tl.sum(probability,axis=0)
    value=tl.load(V+LAYER*VSL+physical[:,None]*VSB+kv_head*VSH+
                  offset[:,None]*VST+dimension[None,:],mask=active[:,None],other=0).to(tl.float32)
    result=tl.where(split_active,tl.sum(probability[:,None]*value,axis=0)/denominator,0.0)
    base=(bh*SPLITS+split)
    tl.store(O+base*D+dimension,result)
    tl.store(LSE+base,tl.where(split_active,safe_max+tl.log(denominator),-float('inf')))


@triton.jit
def _merge(O,LSE,Y,D:tl.constexpr,SPLITS:tl.constexpr,SB:tl.constexpr):
    bh=tl.program_id(0)
    split=tl.arange(0,SB); dimension=tl.arange(0,D)
    logsum=tl.load(LSE+bh*SPLITS+split,mask=split<SPLITS,other=-float('inf'))
    weight=tl.exp(logsum-tl.max(logsum,axis=0))
    partial=tl.load(O+(bh*SPLITS+split[:,None])*D+dimension[None,:],
                    mask=split[:,None]<SPLITS,other=0.0)
    result=tl.sum(weight[:,None]*partial,axis=0)/tl.sum(weight,axis=0)
    tl.store(Y+bh*D+dimension,result)


def paged_attention(q,keys,values,block_tables,lengths,layer,block_size=16,tile=128,
                    *,output=None,partial=None,lse=None):
    """Compute one-query attention without gathering physical pages.

    ``q`` is ``[B,Hq,D]`` and cache storage is
    ``[layer,physical_block,Hkv,block_token,D]``.
    """
    if q.ndim!=3 or keys.ndim!=5 or keys.shape!=values.shape:
        raise ValueError('q[B,Hq,D] and matching paged K/V storage required')
    batch,heads,dimension=q.shape
    if block_tables.ndim!=2 or block_tables.shape[0]!=batch or lengths.shape!=(batch,):
        raise ValueError('block tables [B,max_blocks] and lengths [B] required')
    if keys.shape[3]!=block_size or keys.shape[-1]!=dimension or heads%keys.shape[2]:
        raise ValueError('cache block/head dimensions are incompatible with q')
    if dimension not in (64,128) or tile not in (64,128,256) or block_size&(block_size-1):
        raise ValueError('D=64/128, power-of-two blocks, and tile=64/128/256 required')
    if not 0<=layer<keys.shape[0]: raise IndexError('layer out of range')
    if not q.is_cuda or any(x.device!=q.device for x in (keys,values,block_tables,lengths)):
        raise ValueError('all inputs must share one CUDA device')
    if q.dtype not in (torch.float16,torch.bfloat16) or keys.dtype!=q.dtype or values.dtype!=q.dtype:
        raise ValueError('matching FP16/BF16 q/k/v required')
    if block_tables.dtype!=torch.int32 or lengths.dtype!=torch.int32:
        raise ValueError('block tables and lengths must be int32')
    if q.stride(-1)!=1 or keys.stride(-1)!=1 or values.stride(-1)!=1:
        raise ValueError('unit-stride head dimensions required')
    capacity=block_tables.shape[1]*block_size
    splits=triton.cdiv(capacity,tile)
    if splits>256: raise ValueError('merge supports at most 256 splits')
    output=torch.empty_like(q) if output is None else output
    partial=torch.empty((batch,heads,splits,dimension),device=q.device,dtype=torch.float32) if partial is None else partial
    lse=torch.empty((batch,heads,splits),device=q.device,dtype=torch.float32) if lse is None else lse
    if output.shape!=q.shape or output.dtype!=q.dtype or output.device!=q.device:
        raise ValueError('output workspace mismatch')
    if partial.shape!=(batch,heads,splits,dimension) or partial.dtype!=torch.float32 or partial.device!=q.device:
        raise ValueError('partial workspace mismatch')
    if lse.shape!=(batch,heads,splits) or lse.dtype!=torch.float32 or lse.device!=q.device:
        raise ValueError('LSE workspace mismatch')
    with torch.cuda.device(q.device):
        _partial[(batch*heads,splits)](
            q,keys,values,block_tables,lengths,partial,lse,layer,
            heads,keys.shape[2],dimension,block_size,block_tables.shape[1],splits,tile,
            *keys.stride()[:4],*values.stride()[:4],num_warps=4)
        _merge[(batch*heads,)](partial,lse,output,dimension,splits,
                              triton.next_power_of_2(splits),num_warps=4)
    return output
