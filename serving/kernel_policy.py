"""Capability-based Qwen kernel selection with correctness-first fallbacks."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import torch

DEFAULT_PROFILES=Path(__file__).with_name('calibrations')/'qwen3_0.6b.json'


@dataclass(frozen=True)
class DeviceCapabilities:
    name:str
    major:int
    minor:int
    multiprocessors:int
    warp_size:int
    total_memory:int

    @property
    def architecture(self): return f'sm{self.major}{self.minor}'


@dataclass(frozen=True)
class ModelSignature:
    layers:int
    hidden_size:int
    attention_heads:int
    kv_heads:int
    head_dim:int


@dataclass(frozen=True)
class QwenKernelPolicy:
    profile:str
    architecture:str
    block_size:int
    attention_tile:int
    direct_attention_layers:frozenset[int]
    fused_gather:bool
    native_gqa:bool
    fused_qkv_layers:frozenset[int]
    fused_mlp_layers:frozenset[int]
    calibrated:bool
    reason:str


def detect_cuda_capabilities(device=None):
    device=torch.device('cuda' if device is None else device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise ValueError('CUDA device required')
    properties=torch.cuda.get_device_properties(device)
    major,minor=torch.cuda.get_device_capability(device)
    return DeviceCapabilities(properties.name,major,minor,properties.multi_processor_count,
                              properties.warp_size,properties.total_memory)


def qwen_signature(model):
    config=model.config
    return ModelSignature(len(model.model.layers),config.hidden_size,
                          config.num_attention_heads,config.num_key_value_heads,config.head_dim)


def select_qwen_policy(capabilities,signature,profiles):
    expected=profiles['model_signature']
    actual={
        'layers':signature.layers,'hidden_size':signature.hidden_size,
        'attention_heads':signature.attention_heads,'kv_heads':signature.kv_heads,
        'head_dim':signature.head_dim,
    }
    if actual!=expected:
        return QwenKernelPolicy(
            'fallback',capabilities.architecture,16,128,frozenset(),False,False,
            frozenset(),frozenset(),False,'model signature is not calibrated')
    row=profiles.get('architectures',{}).get(capabilities.architecture)
    if row is None or not row.get('quality_gate_passed',False):
        return QwenKernelPolicy(
            'fallback',capabilities.architecture,16,128,frozenset(),False,False,
            frozenset(),frozenset(),False,
            'GPU architecture has no accepted quality profile')
    layers=frozenset(int(x) for x in row.get('direct_attention_layers',()))
    qkv_layers=frozenset(int(x) for x in row.get('fused_qkv_layers',()))
    mlp_layers=frozenset(int(x) for x in row.get('fused_mlp_layers',()))
    if any(x<0 or x>=signature.layers for x in layers|qkv_layers|mlp_layers):
        raise ValueError('calibration contains an invalid layer index')
    block=int(row.get('block_size',16)); tile=int(row.get('attention_tile',128))
    if block&(block-1) or tile not in (64,128,256):
        raise ValueError('calibration contains an invalid block/tile size')
    return QwenKernelPolicy(
        row.get('name',capabilities.architecture),capabilities.architecture,
        block,tile,layers,bool(row.get('fused_gather',False)),
        bool(row.get('native_gqa',False)),qkv_layers,mlp_layers,True,
        'matched accepted architecture profile')


def load_qwen_policy(model,device=None,path=DEFAULT_PROFILES):
    profiles=json.loads(Path(path).read_text(encoding='utf-8'))
    return select_qwen_policy(detect_cuda_capabilities(device),qwen_signature(model),profiles)
