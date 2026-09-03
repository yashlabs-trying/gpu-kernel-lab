"""Opt-in W4A16 single-token A/B experiment; retains dense prefill weights."""
import sys
import types
from pathlib import Path

import torch


def install_int4_decode(model, *, lm_head_only=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'learning/09_int4'))
    from quantization import quantize_int4
    from kernels import int4_gemv
    if getattr(model, '_kernellab_int4_installed', False):
        raise ValueError('INT4 ablation already installed')
    state = {'single_token': False}

    def detect_decode(module, positional, keywords):
        ids = keywords.get('input_ids')
        if ids is None and positional:
            ids = positional[0]
        state['single_token'] = ids is not None and ids.ndim == 2 and tuple(ids.shape) == (1,1)

    count, packed_bytes = 0, 0
    for name, module in model.named_modules():
        if lm_head_only and name != 'lm_head':
            continue
        if isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                raise ValueError('this Qwen ablation supports bias-free projections only')
            packed = quantize_int4(module.weight)
            original = module.forward

            def forward(self, x, packed=packed, original=original):
                if not state['single_token'] or x.numel() != x.shape[-1]:
                    return original(x)
                y = int4_gemv(x.reshape(-1), packed)
                return y.reshape(*x.shape[:-1], packed.shape[0])

            module.forward = types.MethodType(forward, module)
            count += 1
            packed_bytes += packed.storage_bytes
    model.register_forward_pre_hook(detect_decode, with_kwargs=True)
    model._kernellab_int4_installed = True
    model._kernellab_int4_info = {
        'packed_bytes_added':packed_bytes, 'linear_modules':count, 'lm_head_only':lm_head_only,
        'warning':'Dense weights retained for prefill; GPU memory increases in this A/B adapter. Full logits retained for Transformers generation; compact top-k is tested separately.'}
    return count
