"""AWQ-inspired W8A16 channel-scale search, NOT a full AWQ implementation.

No activation quantization, clipping search, or checkpoint mutation. Calibration
minimizes local projection output MSE; held-out model quality remains mandatory.
"""
from dataclasses import dataclass
import torch


@dataclass
class CalibratedInt8:
    data: torch.Tensor
    scales: torch.Tensor
    inverse: torch.Tensor
    group_size: int
    alpha: float
    calibration_mse: float

    @property
    def storage_bytes(self):
        return self.data.numel() + 4*(self.scales.numel()+self.inverse.numel())


@torch.inference_mode()
def quantize_int8(weight, calibration, group_size=128, alphas=(0., .5, 1.)):
    if weight.ndim != 2 or calibration.ndim != 2 or weight.shape[1] != calibration.shape[1]:
        raise ValueError('expected weight[N,K], calibration[T,K]')
    n, k = weight.shape
    if n == 0 or k == 0 or group_size <= 0 or k % group_size or calibration.shape[0] == 0:
        raise ValueError('nonempty tensors and group dividing K required')
    if not alphas or any(a < 0 or a > 1 for a in alphas):
        raise ValueError('nonempty alpha values in [0,1] required')
    if weight.device != calibration.device or not weight.is_floating_point() or not calibration.is_floating_point():
        raise ValueError('colocated floating-point inputs required')
    if not torch.isfinite(weight).all().item() or not torch.isfinite(calibration).all().item():
        raise ValueError('finite inputs required')
    x = calibration.float()
    importance = x.abs().mean(0).clamp_min(1e-4)
    best = None
    # FP32 reference, with TF32 disabled by the calling evaluation harness.
    for alpha in alphas:
        channel = importance.pow(alpha).clamp(1e-2, 1e2)
        channel /= (channel.max()*channel.min()).sqrt()
        data = torch.empty((n,k), dtype=torch.int8, device=weight.device)
        scales = torch.empty((n,k//group_size), dtype=torch.float32, device=weight.device)
        error = 0.
        for start in range(0,n,256):
            w = weight[start:start+256].float()
            grouped = (w*channel).reshape(w.shape[0],-1,group_size)
            scale = grouped.abs().amax(-1)/127
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = (grouped/scale[...,None]).round().clamp(-127,127).to(torch.int8)
            restored = (q.float()*scale[...,None]).reshape(w.shape)/channel
            error += ((x@restored.T - x@w.T).square().sum()).item()
            data[start:start+w.shape[0]] = q.reshape(w.shape)
            scales[start:start+w.shape[0]] = scale
        candidate = CalibratedInt8(data,scales,channel.reciprocal().contiguous(),group_size,float(alpha),error/(x.shape[0]*n))
        if best is None or candidate.calibration_mse < best.calibration_mse:
            best = candidate
    return best


def dequantize_int8(weight):
    return weight.data.float()*weight.scales.repeat_interleave(weight.group_size,1)*weight.inverse
