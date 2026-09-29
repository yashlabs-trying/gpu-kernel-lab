"""Read-only Qwen deployment preflight for heterogeneous CUDA hosts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from .kernel_policy import (
    DEFAULT_PROFILES,ModelSignature,detect_cuda_capabilities,select_qwen_policy,
)


def signature_from_config(config):
    return ModelSignature(
        int(config.num_hidden_layers),int(config.hidden_size),
        int(config.num_attention_heads),int(config.num_key_value_heads),
        int(config.head_dim),
    )


def inspect_qwen_host(config,profiles_path=DEFAULT_PROFILES):
    """Return a serializable deployment decision without allocating model weights."""
    profiles=json.loads(Path(profiles_path).read_text(encoding='utf-8'))
    signature=signature_from_config(config)
    devices=[]
    for index in range(torch.cuda.device_count()):
        capabilities=detect_cuda_capabilities(torch.device('cuda',index))
        policy=select_qwen_policy(capabilities,signature,profiles)
        capability_data={**asdict(capabilities),'architecture':capabilities.architecture}
        devices.append({
            'index':index,
            'capabilities':capability_data,
            'policy':{
                **asdict(policy),
                'direct_attention_layers':sorted(policy.direct_attention_layers),
                'fused_qkv_layers':sorted(policy.fused_qkv_layers),
                'fused_mlp_layers':sorted(policy.fused_mlp_layers),
            },
        })
    peers=[
        [i==j or bool(torch.cuda.can_device_access_peer(i,j))
         for j in range(len(devices))]
        for i in range(len(devices))
    ]
    homogeneous=len({x['capabilities']['architecture'] for x in devices})<=1
    full_peer=all(all(row) for row in peers)
    if not devices:
        recommendation='cuda_required'
    elif len(devices)==1:
        recommendation='single_gpu'
    elif homogeneous and full_peer:
        recommendation='tensor_parallel_candidate'
    else:
        recommendation='one_process_per_gpu'
    return {
        'model_signature':asdict(signature),
        'cuda_available':bool(devices),
        'device_count':len(devices),
        'devices':devices,
        'peer_access':peers,
        'homogeneous_architectures':homogeneous,
        'topology_recommendation':recommendation,
        'safe_to_start':bool(devices),
        'note':(
            'Uncalibrated devices use exact kernels. Tensor parallel execution '
            'requires a separately validated distributed launcher.'),
    }
