"""Correctness-first Qwen executor backed by :class:`PagedKVAllocator`.

This closes the serving/model contract without pretending that cache gathering is
the final kernel. K/V are written once into physical pages and all attention
reads are resolved through the request block table. A later Triton/CUDA kernel
can replace ``materialize_paged_kv`` without changing the scheduler or API.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def materialize_paged_kv(storage,layer,block_table,length,block_size):
    """Return logical ``[Hkv,S,D]`` order from physical paged storage."""
    if length<=0:
        raise ValueError('attention length must be positive')
    blocks=math.ceil(length/block_size)
    if block_table.ndim!=1 or block_table.numel()<blocks:
        raise ValueError('block table is too short')
    physical=block_table[:blocks].to(torch.long)
    # CUDA metadata is allocator-produced; avoid a device synchronization in
    # every layer. CPU reference calls still reject malformed tables eagerly.
    if physical.device.type=='cpu' and bool((physical<0).any()):
        raise ValueError('block table contains an unallocated entry')
    # [blocks,Hkv,block,D] -> [Hkv,logical_tokens,D]
    return storage[layer,physical].permute(1,0,2,3).reshape(
        storage.shape[2],blocks*block_size,storage.shape[-1])[:,:length]


def _rotate_half(x):
    half=x.shape[-1]//2
    return torch.cat((-x[...,half:],x[...,:half]),dim=-1)


class QwenPagedExecutor:
    """Real Qwen forward path using scheduler-owned physical K/V pages.

    The first implementation deliberately executes requests serially. Prefill
    keeps Qwen's tiled SDPA math and scatters new cache ranges into pages. This
    establishes exact token/cache semantics before direct paged attention and
    mixed-batch kernels are enabled.
    """
    def __init__(self,model,allocator,*,direct_attention=False,direct_attention_layers=None,
                 attention_tile=128,policy=None,fused_gather=True):
        self.model=model.eval(); self.allocator=allocator
        self.device=next(model.parameters()).device
        self.dtype=next(model.parameters()).dtype
        config=model.config
        expected=(len(model.model.layers),config.num_key_value_heads,config.head_dim)
        actual=(allocator.num_layers,allocator.num_kv_heads,allocator.head_dim)
        if expected!=actual:
            raise ValueError(f'model/cache shape mismatch: expected {expected}, got {actual}')
        if allocator.keys.device!=self.device or allocator.keys.dtype!=self.dtype:
            raise ValueError('model and paged cache must share device and dtype')
        if self.device.type!='cuda':
            raise ValueError('QwenPagedExecutor requires a CUDA model')
        self._prefill_caches={}
        if policy is not None:
            if direct_attention or direct_attention_layers is not None:
                raise ValueError('explicit direct-attention settings conflict with policy')
            direct_attention_layers=policy.direct_attention_layers
            attention_tile=policy.attention_tile
            if allocator.block_size!=policy.block_size:
                raise ValueError('allocator block size does not match selected policy')
        if attention_tile not in (64,128,256): raise ValueError('attention tile must be 64, 128, or 256')
        self.policy=policy; self.attention_tile=attention_tile; self.fused_gather=bool(fused_gather)
        layer_count=len(model.model.layers)
        if direct_attention_layers is None:
            self.direct_attention_layers=(frozenset(range(layer_count)) if direct_attention else frozenset())
        else:
            selected=frozenset(int(x) for x in direct_attention_layers)
            if any(x<0 or x>=layer_count for x in selected):
                raise ValueError('direct attention layer index out of range')
            self.direct_attention_layers=selected
        self._attention_workspaces={}
        self._gather_workspaces={}

    async def prefill(self,work,allocator):
        self._check_allocator(allocator)
        logits={}
        with torch.inference_mode():
            for item in work:
                token=torch.tensor([item.token_ids],device=self.device,dtype=torch.long)
                mask=torch.ones((1,item.start_position+len(item.token_ids)),
                                device=self.device,dtype=torch.long)
                output=self.model(
                    input_ids=token,attention_mask=mask,
                    past_key_values=self._prefill_caches.get(item.request_id),
                    use_cache=True,logits_to_keep=1)
                self._prefill_caches[item.request_id]=output.past_key_values
                self._scatter_prefill(item,output.past_key_values)
                logits[item.request_id]=output.logits[0,-1].float()
        return logits

    async def decode(self,work,metadata,graph_batch_size):
        if metadata.request_ids!=tuple(item.request_id for item in work):
            raise ValueError('decode metadata order does not match work order')
        with torch.inference_mode():
            for row,item in enumerate(work):
                self._prefill_caches.pop(item.request_id,None)
                if int(metadata.lengths[row].item())!=item.position+1:
                    raise ValueError('decode cache length must include exactly the new slot')
            output=self._forward_batch(work,metadata.block_tables,metadata.lengths)
        return {item.request_id:output[row] for row,item in enumerate(work)}

    def release(self,request_id):
        """Release executor-private temporary state after any terminal event."""
        self._prefill_caches.pop(request_id,None)

    def _scatter_prefill(self,item,cache):
        start=item.start_position; end=start+len(item.token_ids)
        for layer_index,layer_cache in enumerate(cache.layers):
            keys=layer_cache.keys[0,:,start:end]
            values=layer_cache.values[0,:,start:end]
            for source,(block,offset) in enumerate(item.slots):
                self.allocator.keys[layer_index,block,:,offset].copy_(keys[:,source])
                self.allocator.values[layer_index,block,:,offset].copy_(values[:,source])

    def _check_allocator(self,allocator):
        if allocator is not self.allocator:
            raise ValueError('executor received a different paged allocator')

    def _forward_batch(self,work,block_tables,lengths):
        batch=len(work); host_lengths=lengths.tolist(); maximum=max(host_lengths)
        needs_mask=any(length!=maximum for length in host_lengths)
        for row,item in enumerate(work):
            self._validate_slot(item.position,item.slot,block_tables[row])
        token=torch.tensor([[item.token_id] for item in work],device=self.device,dtype=torch.long)
        position_ids=torch.tensor([[item.position] for item in work],device=self.device,dtype=torch.long)
        hidden=self.model.model.embed_tokens(token)
        cos,sin=self.model.model.rotary_emb(hidden,position_ids)
        cos=cos[:,None,:,:]; sin=sin[:,None,:,:]

        for layer_index,layer in enumerate(self.model.model.layers):
            residual=hidden
            normalized=layer.input_layernorm(hidden)
            attention=layer.self_attn
            q=attention.q_norm(attention.q_proj(normalized).view(
                batch,1,attention.config.num_attention_heads,attention.head_dim)).transpose(1,2)
            k=attention.k_norm(attention.k_proj(normalized).view(
                batch,1,attention.config.num_key_value_heads,attention.head_dim)).transpose(1,2)
            v=attention.v_proj(normalized).view(
                batch,1,attention.config.num_key_value_heads,attention.head_dim).transpose(1,2)
            q=q*cos+_rotate_half(q)*sin
            k=k*cos+_rotate_half(k)*sin
            for row,item in enumerate(work):
                block,block_offset=item.slot
                self.allocator.keys[layer_index,block,:,block_offset].copy_(k[row,:,0])
                self.allocator.values[layer_index,block,:,block_offset].copy_(v[row,:,0])
            if layer_index in self.direct_attention_layers:
                output=self._paged_attention(q[:,:,0],block_tables,lengths,layer_index)[:,:,None]
            elif self.fused_gather:
                keys,values=self._gather_attention(block_tables,lengths,layer_index,maximum)
                mask=None
                if needs_mask:
                    positions=torch.arange(maximum,device=self.device)
                    valid=positions[None,:]<lengths[:,None]
                    mask=torch.zeros((batch,1,1,maximum),device=self.device,dtype=self.dtype)
                    mask.masked_fill_(~valid[:,None,None,:],torch.finfo(self.dtype).min)
                output=F.scaled_dot_product_attention(
                    q,keys,values,attn_mask=mask,dropout_p=0.0,is_causal=False,
                    scale=getattr(attention,'scaling',attention.head_dim**-0.5))
            else:
                gathered_keys=[]; gathered_values=[]
                for row,length in enumerate(host_lengths):
                    keys=materialize_paged_kv(self.allocator.keys,layer_index,block_tables[row],
                                              length,self.allocator.block_size)
                    values=materialize_paged_kv(self.allocator.values,layer_index,block_tables[row],
                                                length,self.allocator.block_size)
                    gathered_keys.append(F.pad(keys,(0,0,0,maximum-length)))
                    gathered_values.append(F.pad(values,(0,0,0,maximum-length)))
                keys=torch.stack(gathered_keys); values=torch.stack(gathered_values)
                # Match Transformers' explicit Qwen GQA expansion ordering.
                groups=attention.config.num_attention_heads//attention.config.num_key_value_heads
                keys=keys.repeat_interleave(groups,dim=1)
                values=values.repeat_interleave(groups,dim=1)
                mask=None
                if needs_mask:
                    positions=torch.arange(maximum,device=self.device)
                    valid=positions[None,:]<lengths[:,None]
                    mask=torch.zeros((len(work),1,1,maximum),device=self.device,dtype=self.dtype)
                    mask.masked_fill_(~valid[:,None,None,:],torch.finfo(self.dtype).min)
                output=F.scaled_dot_product_attention(
                    q,keys,values,attn_mask=mask,dropout_p=0.0,is_causal=False,
                    scale=getattr(attention,'scaling',attention.head_dim**-0.5))
            output=output.transpose(1,2).reshape(batch,1,-1)
            hidden=residual+attention.o_proj(output)
            residual=hidden
            hidden=residual+layer.mlp(layer.post_attention_layernorm(hidden))

        hidden=self.model.model.norm(hidden)
        return self.model.lm_head(hidden)[:,0].float()

    def _gather_attention(self,block_tables,lengths,layer,maximum):
        from .paged_gather import gather_paged_gqa
        batch,width=block_tables.shape; capacity=width*self.allocator.block_size
        key=(batch,width); workspace=self._gather_workspaces.get(key)
        if workspace is None:
            shape=(batch,self.model.config.num_attention_heads,capacity,self.model.config.head_dim)
            workspace=(torch.empty(shape,device=self.device,dtype=self.dtype),
                       torch.empty(shape,device=self.device,dtype=self.dtype))
            self._gather_workspaces[key]=workspace
        gather_paged_gqa(self.allocator.keys,self.allocator.values,block_tables,lengths,
                         layer,self.allocator.block_size,key_output=workspace[0],value_output=workspace[1])
        return workspace[0][:,:,:maximum],workspace[1][:,:,:maximum]

    def _paged_attention(self,q,block_tables,lengths,layer):
        from .paged_attention import paged_attention
        import triton
        batch,width=block_tables.shape; splits=triton.cdiv(width*self.allocator.block_size,self.attention_tile)
        key=(batch,width); workspace=self._attention_workspaces.get(key)
        if workspace is None:
            heads,dimension=q.shape[1:]
            workspace=(
                torch.empty_like(q),
                torch.empty((batch,heads,splits,dimension),device=q.device,dtype=torch.float32),
                torch.empty((batch,heads,splits),device=q.device,dtype=torch.float32),
            )
            self._attention_workspaces[key]=workspace
        return paged_attention(
            q,self.allocator.keys,self.allocator.values,block_tables,lengths,
            layer,self.allocator.block_size,self.attention_tile,
            output=workspace[0],partial=workspace[1],lse=workspace[2])

    def _validate_slot(self,position,slot,block_table):
        logical_block,offset=divmod(position,self.allocator.block_size)
        if logical_block>=block_table.numel():
            raise ValueError('position is outside the block table')
        expected=(int(block_table[logical_block].item()),offset)
        if tuple(slot)!=expected:
            raise ValueError(f'physical slot mismatch: expected {expected}, got {slot}')
