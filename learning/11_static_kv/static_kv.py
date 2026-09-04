"""Qwen3 decode integration for the fused Q/K-normalize/RoPE/cache-write kernel.

Restricted to eager, B=1, one-token decode with a Transformers StaticCache.
Prefill and unsupported calls use the original Transformers implementation.
"""
import sys
import types
from pathlib import Path
import torch
import triton
import triton.language as tl
from transformers.cache_utils import StaticCache
from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'learning/08_decode'))
from fused_qkv_cache import fused_qk_norm_rope_cache


@triton.jit
def advance_position_kernel(POSITION):
    # A fixed-address scalar replaces arange/allocation on every decode step.
    value=tl.load(POSITION)
    tl.store(POSITION,value+1)


@triton.jit
def open_mask_kernel(MASK,POSITION,CAPACITY:tl.constexpr):
    position=tl.load(POSITION)
    tl.store(MASK+position,0.0,mask=(position>=0)&(position<CAPACITY))


def advance_position(position):
    advance_position_kernel[(1,)](position,num_warps=1)


def open_mask(mask,position):
    open_mask_kernel[(1,)](mask,position,mask.shape[-1],num_warps=1)


def make_mask(batch,capacity,valid,dtype,device):
    if not 0 <= valid <= capacity:
        raise ValueError('valid prefix must fit capacity')
    mask=torch.full((batch,1,1,capacity),torch.finfo(dtype).min,dtype=dtype,device=device)
    mask[..., :valid]=0
    return mask


@torch.inference_mode()
def install_fused_static_kv(model,capacity,batch_size=1):
    if getattr(model,'_kernellab_static_kv',None) is not None:
        raise ValueError('static KV integration already installed')
    config=model.config
    if batch_size != 1 or capacity <= 0 or config.head_dim != 128 or config.num_attention_heads != 16 or config.num_key_value_heads != 8:
        raise ValueError('this Qwen3-0.6B experiment requires B=1,Hq=16,Hkv=8,D=128')
    if model.device.type!='cuda' or next(model.parameters()).dtype != torch.bfloat16 or model.training:
        raise ValueError('eval-mode CUDA BF16 model required')
    device=model.device
    original_rotary=model.model.rotary_emb.forward
    positions=torch.arange(capacity,device=device).unsqueeze(0)
    dummy=torch.empty((batch_size,1,config.hidden_size),device=device,dtype=torch.bfloat16)
    full_cos,full_sin=original_rotary(dummy,positions)
    # Qwen repeats the half-width frequencies. The fused kernel reconstructs
    # both halves from this compact table.
    cos=full_cos[0,:,:config.head_dim//2].contiguous()
    sin=full_sin[0,:,:config.head_dim//2].contiguous()
    del positions,full_cos,full_sin
    rotary_dummy=torch.empty((batch_size,1,config.head_dim),device=device,dtype=torch.bfloat16)
    state={'decode':False}

    def rotary(self,x,position_ids):
        if state['decode'] and x.shape[-2]==1:
            return rotary_dummy,rotary_dummy
        return original_rotary(x,position_ids)
    model.model.rotary_emb.forward=types.MethodType(rotary,model.model.rotary_emb)

    for attention in (layer.self_attn for layer in model.model.layers):
        original=attention.forward
        def forward(self,hidden_states,position_embeddings,attention_mask,
                    past_key_values=None,original=original,**kwargs):
            if not (state['decode'] and hidden_states.shape[:2]==(1,1) and isinstance(past_key_values,StaticCache)):
                return original(hidden_states,position_embeddings,attention_mask,past_key_values,**kwargs)
            if not isinstance(attention_mask,torch.Tensor) or attention_mask.shape != (1,1,1,capacity):
                raise ValueError('decode requires the persistent additive mask')
            position_ids=kwargs.get('position_ids')
            if position_ids is None or position_ids.shape!=(1,1) or not position_ids.is_contiguous():
                raise ValueError('fixed contiguous position_ids [1,1] required')
            layer=past_key_values.layers[self.layer_idx]
            if not layer.is_initialized:
                raise ValueError('StaticCache must be initialized by prefill')
            q=self.q_proj(hidden_states).view(1,1,self.config.num_attention_heads,self.head_dim)[:,0]
            k=self.k_proj(hidden_states).view(1,1,self.config.num_key_value_heads,self.head_dim)[:,0]
            v=self.v_proj(hidden_states).view(1,1,self.config.num_key_value_heads,self.head_dim)[:,0]
            q=fused_qk_norm_rope_cache(q,k,v,self.q_norm.weight,self.k_norm.weight,
                cos,sin,position_ids[:,0],layer.keys,layer.values,attention_mask,
                epsilon=self.q_norm.variance_epsilon)
            interface=ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,eager_attention_forward)
            output,weights=interface(self,q.unsqueeze(2),layer.keys,layer.values,attention_mask,
                dropout=0.0,scaling=self.scaling,sliding_window=self.sliding_window,**kwargs)
            output=output.reshape(*hidden_states.shape[:-1],-1).contiguous()
            return self.o_proj(output),weights
        attention.forward=types.MethodType(forward,attention)

    def detect(module,positional,keywords):
        ids=keywords.get('input_ids',positional[0] if positional else None)
        state['decode']=ids is not None and tuple(ids.shape)==(1,1)
    model.register_forward_pre_hook(detect,with_kwargs=True)
    model._kernellab_static_kv={'capacity':capacity,'batch_size':batch_size,
        'cos':cos,'sin':sin,'state':state,
        'limitations':'eager B1 one-token decode; explicit fixed position/mask; no generate/concurrent forwards; cache length metadata stays at prefill length'}
    return model


def new_static_cache(model,capacity):
    return StaticCache(config=model.config,max_cache_len=capacity)
