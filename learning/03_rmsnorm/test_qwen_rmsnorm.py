"""Tolerance checks, not bitwise-equivalence certification."""
import pytest
import torch

from qwen_rmsnorm import (
    qwen_residual_rmsnorm_reference, qwen_rmsnorm_reference,
    triton_qwen_residual_rmsnorm, triton_qwen_rmsnorm,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(1, 128), (33, 128), (1, 1024), (17, 1024), (3, 127)])
@pytest.mark.parametrize('scale', [0., 1e-4, 1., 100.])
@pytest.mark.parametrize('warps', [2, 4, 8])
def test_qwen_reference(dtype, shape, scale, warps):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    torch.manual_seed(42)
    x = torch.randn(shape, device='cuda', dtype=dtype) * scale
    module = Qwen3RMSNorm(shape[-1], eps=1e-6).to(device='cuda', dtype=dtype)
    with torch.inference_mode():
        module.weight.copy_(torch.randn_like(module.weight))
        expected = module(x)
        torch.testing.assert_close(qwen_rmsnorm_reference(x, module.weight), expected, rtol=0, atol=0)
        actual = triton_qwen_rmsnorm(x, module.weight, num_warps=warps)
    tolerance = {torch.float32: (2e-5, 2e-6), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}
    rtol, atol = tolerance[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_reject_noncontiguous():
    x = torch.randn(128, 4, device='cuda').T
    with pytest.raises(ValueError, match='contiguous'):
        triton_qwen_rmsnorm(x, torch.ones(128, device='cuda'))


def test_empty_rows():
    x = torch.empty(0, 128, device='cuda')
    assert triton_qwen_rmsnorm(x, torch.ones(128, device='cuda')).shape == x.shape


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(1, 1024), (257, 1024)])
def test_qwen_residual_add_and_norm(dtype, shape):
    torch.manual_seed(7)
    x = torch.randn(shape, device='cuda', dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], device='cuda', dtype=dtype)
    with torch.inference_mode():
        expected_y, expected_residual = qwen_residual_rmsnorm_reference(x, residual, weight)
        actual_y, actual_residual = triton_qwen_residual_rmsnorm(x, residual, weight)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_y, expected_y, rtol=2e-2, atol=2e-2)
