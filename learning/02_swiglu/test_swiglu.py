"""Correctness and boundary tests for fused SwiGLU."""

import pytest
import torch

from swiglu import pytorch_swiglu, triton_swiglu


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 2e-3, "atol": 2e-3},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


@requires_cuda
@pytest.mark.parametrize("size", [1, 7, 127, 128, 129, 1_000, 100_003])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matches_pytorch(size: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    gate = torch.randn(size, device="cuda", dtype=dtype)
    up = torch.randn(size, device="cuda", dtype=dtype)

    expected = pytorch_swiglu(gate, up)
    actual = triton_swiglu(gate, up)
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


@requires_cuda
@pytest.mark.parametrize("block_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("num_warps", [2, 4, 8])
def test_all_tuning_configurations(block_size: int, num_warps: int) -> None:
    # 1,003 deliberately crosses every tested block boundary irregularly.
    gate = torch.randn(1_003, device="cuda")
    up = torch.randn(1_003, device="cuda")
    torch.testing.assert_close(
        triton_swiglu(gate, up, block_size, num_warps),
        pytorch_swiglu(gate, up),
        **TOLERANCES[torch.float32],
    )

