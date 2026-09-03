"""Experimental BF16/W8A16 warp-per-row CUDA. No default model replacement."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from torch.utils.cpp_extension import load


@lru_cache(None)
def extension():
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='kernellab_warp_decode_v1',
                sources=[str(Path(__file__).with_name('warp_gemv.cu'))],
                extra_cuda_cflags=['-O3', '-lineinfo'], verbose=False)


def project(x, weight, *, warps=4, persistent=False, swiglu=False):
    """weight is BF16 [N,K] or CalibratedInt8; gate rows precede up rows."""
    if torch.is_grad_enabled() and x.requires_grad:
        raise ValueError('inference only')
    if isinstance(weight, torch.Tensor):
        # Unused tensors avoid allocation and are not dereferenced by BF16 path.
        return extension().project(x, weight, x, x, 128, warps, persistent, swiglu)
    return extension().project(x, weight.data, weight.scales, weight.inverse,
                               weight.group_size, warps, persistent, swiglu)
