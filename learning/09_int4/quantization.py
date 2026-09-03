"""Offline symmetric groupwise INT4 weights. No checkpoint is modified."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedInt4:
    data: torch.Tensor       # uint8 [N,K/2], low nibble precedes high nibble
    scales: torch.Tensor     # FP32 [N,K/group_size]
    group_size: int

    @property
    def shape(self):
        return (self.data.shape[0], self.data.shape[1] * 2)

    @property
    def storage_bytes(self):
        return self.data.numel() + self.scales.numel() * 4

    def validate(self):
        if self.group_size not in (32, 64, 128):
            raise ValueError('group_size must be 32, 64, or 128')
        if self.data.ndim != 2 or self.data.dtype != torch.uint8:
            raise ValueError('packed data must be uint8 [N,K/2]')
        n, k = self.shape
        if n == 0 or k == 0 or k % self.group_size:
            raise ValueError('positive dimensions and K divisible by group_size required')
        if self.scales.shape != (n, k // self.group_size) or self.scales.dtype != torch.float32:
            raise ValueError('scales must be FP32 [N,K/group_size]')
        if self.data.device != self.scales.device or not self.data.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError('contiguous colocated buffers required')


@torch.no_grad()
def quantize_int4(weight, group_size=128, chunk_rows=256):
    """Absmax/7, nearest-even rounding, [-7,7], signed two's-complement nibbles.

    Row chunks bound temporary FP32 memory; the returned object retains no
    original weights. This is simple RTN quantization, not calibrated GPTQ/AWQ.
    """
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('weight must be floating-point [N,K]')
    n, k = weight.shape
    if group_size not in (32, 64, 128) or n == 0 or k == 0 or k % group_size:
        raise ValueError('positive dimensions; group 32/64/128 must divide K')
    if chunk_rows <= 0:
        raise ValueError('chunk_rows must be positive')
    data = torch.empty((n, k // 2), device=weight.device, dtype=torch.uint8)
    scales = torch.empty((n, k // group_size), device=weight.device, dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        block = weight[start:start + chunk_rows].float()
        if not torch.isfinite(block).all().item():
            raise ValueError('weights must be finite')
        grouped = block.reshape(block.shape[0], k // group_size, group_size)
        scale = grouped.abs().amax(-1) / 7
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quant = (grouped / scale[..., None]).round().clamp(-7, 7).to(torch.int16).reshape(block.shape)
        codes = quant & 15
        data[start:start + block.shape[0]] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        scales[start:start + block.shape[0]] = scale
    return PackedInt4(data, scales, group_size)


def dequantize_int4(packed):
    """Full FP32 reference ONLY; never called by the fused decode path."""
    packed.validate()
    low, high = packed.data & 15, packed.data >> 4
    codes = torch.stack((low, high), -1).reshape(packed.shape).to(torch.int16)
    signed = torch.where(codes >= 8, codes - 16, codes).float()
    return signed * packed.scales.repeat_interleave(packed.group_size, dim=1)
