"""Exact-shape Qwen prefill GEMM dispatch; every other shape stays on PyTorch."""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from qwen_prefill_gemm import qwen_prefill_linear


# (module role, M, K, N). These are the only RTX-3090 shapes that cleared the
# prior isolated benchmark by a meaningful margin selected for integration.
WINNERS={
    ('q_proj',2048,1024,2048),
}

# These won isolated GEMMs but lost after model integration, so the dispatcher
# deliberately does not activate them.
REJECTED_AFTER_INTEGRATION={
    ('k_proj',4096,1024,1024),('v_proj',4096,1024,1024),
    ('gate_proj',512,1024,3072),('up_proj',512,1024,3072),
    ('down_proj',4096,3072,1024),
}


@torch.inference_mode()
def install_prefill_gemm_dispatch(model):
    """Install a strict whitelist; decode and non-winning prefill use nn.Linear."""
    if getattr(model,'_kernellab_prefill_dispatch',None) is not None:
        raise ValueError('prefill GEMM dispatcher already installed')
    installed=[]
    for name,module in model.named_modules():
        role=name.rsplit('.',1)[-1]
        candidates=[shape for shape in WINNERS if shape[0]==role]
        if not candidates or not isinstance(module,torch.nn.Linear) or module.bias is not None:
            continue
        original=module.forward
        def forward(self,x,role=role,original=original):
            k=x.shape[-1]; m=x.numel()//k; n=self.out_features
            if (role,m,k,n) not in WINNERS or x.dtype!=self.weight.dtype or not x.is_contiguous():
                return original(x)
            return qwen_prefill_linear(x,self.weight)
        module.forward=types.MethodType(forward,module)
        installed.append(name)
    model._kernellab_prefill_dispatch={
        'modules':installed,'winners':sorted(WINNERS),
        'rejected_after_integration':sorted(REJECTED_AFTER_INTEGRATION),
        'extra_weight_bytes':0,
        'policy':'exact-shape whitelist; PyTorch fallback for all other shapes',
    }
    return len(installed)
