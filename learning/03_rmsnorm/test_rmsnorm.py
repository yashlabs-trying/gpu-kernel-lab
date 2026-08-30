"""Accuracy, shape, masking, and fusion tests for RMSNorm."""

import pytest
import torch

from rmsnorm import (
    pytorch_residual_rmsnorm, pytorch_rmsnorm,
    triton_residual_rmsnorm, triton_rmsnorm,
)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
TOLERANCES = {
    torch.float32: {"rtol": 2e-5, "atol": 2e-6},
    torch.float16: {"rtol": 2e-3, "atol": 2e-3},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 7), (3, 127), (8, 128), (5, 1_000), (4, 1_024), (2, 4_096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_standalone_matches_pytorch(shape, dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    torch.testing.assert_close(
        triton_rmsnorm(x, weight), pytorch_rmsnorm(x, weight, 1e-6),
        **TOLERANCES[dtype],
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_residual_matches_pytorch(dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn((7, 1_000), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(1_000, device="cuda", dtype=dtype)
    actual_norm, actual_residual = triton_residual_rmsnorm(x, residual, weight)
    expected_norm, expected_residual = pytorch_residual_rmsnorm(x, residual, weight, 1e-6)
    torch.testing.assert_close(actual_norm, expected_norm, **TOLERANCES[dtype])
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)

