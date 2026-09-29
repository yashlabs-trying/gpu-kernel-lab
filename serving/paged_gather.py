"""Lossless fused physical-page gather and GQA expansion for SDPA."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(K,V,TABLE,LENGTHS,KO,VO,LAYER,
            H:tl.constexpr,HKV:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr,
            WIDTH:tl.constexpr,CAPACITY:tl.constexpr,TILE:tl.constexpr,
            KSL:tl.constexpr,KSB:tl.constexpr,KSH:tl.constexpr,KST:tl.constexpr,
            VSL:tl.constexpr,VSB:tl.constexpr,VSH:tl.constexpr,VST:tl.constexpr):
    bh,tile_id=tl.program_id(0),tl.program_id(1)
    batch,head=bh//H,bh%H
    token=tile_id*TILE+tl.arange(0,TILE)
    dimension=tl.arange(0,D)
    valid=token<tl.load(LENGTHS+batch)
    logical=token//BLOCK
    physical=tl.load(TABLE+batch*WIDTH+logical,mask=valid,other=0)
    offset=token%BLOCK; kv_head=head//(H//HKV)
    key=tl.load(K+LAYER*KSL+physical[:,None]*KSB+kv_head*KSH+
                offset[:,None]*KST+dimension[None,:],mask=valid[:,None],other=0)
    value=tl.load(V+LAYER*VSL+physical[:,None]*VSB+kv_head*VSH+
                  offset[:,None]*VST+dimension[None,:],mask=valid[:,None],other=0)
    output=(bh*CAPACITY+token[:,None])*D+dimension[None,:]
    tl.store(KO+output,key,mask=token[:,None]<CAPACITY)
    tl.store(VO+output,value,mask=token[:,None]<CAPACITY)


def gather_paged_gqa(keys,values,block_tables,lengths,layer,block_size=16,*,
                     key_output=None,value_output=None,tile=16):
    if keys.ndim!=5 or keys.shape!=values.shape or block_tables.ndim!=2:
        raise ValueError('paged K/V and block tables [B,width] required')
    batch=block_tables.shape[0]; capacity=block_tables.shape[1]*block_size
    if lengths.shape!=(batch,) or lengths.dtype!=torch.int32 or block_tables.dtype!=torch.int32:
        raise ValueError('int32 lengths [B] and block tables required')
    if not keys.is_cuda or any(x.device!=keys.device for x in (values,block_tables,lengths)):
        raise ValueError('all inputs must share one CUDA device')
    if keys.dtype not in (torch.float16,torch.bfloat16) or values.dtype!=keys.dtype:
        raise ValueError('matching FP16/BF16 cache required')
    if not 0<=layer<keys.shape[0] or keys.shape[3]!=block_size:
        raise ValueError('layer/block mismatch')
    heads=keys.shape[2]*2; dimension=keys.shape[-1]
    shape=(batch,heads,capacity,dimension)
    key_output=torch.empty(shape,device=keys.device,dtype=keys.dtype) if key_output is None else key_output
    value_output=torch.empty(shape,device=keys.device,dtype=keys.dtype) if value_output is None else value_output
    if key_output.shape!=shape or value_output.shape!=shape or key_output.dtype!=keys.dtype or value_output.dtype!=keys.dtype:
        raise ValueError('gather workspace mismatch')
    if not key_output.is_contiguous() or not value_output.is_contiguous():
        raise ValueError('contiguous gather workspace required')
    with torch.cuda.device(keys.device):
        _gather[(batch*heads,triton.cdiv(capacity,tile))](
            keys,values,block_tables,lengths,key_output,value_output,layer,
            heads,keys.shape[2],dimension,block_size,block_tables.shape[1],capacity,tile,
            *keys.stride()[:4],*values.stride()[:4],num_warps=4)
    return key_output,value_output
