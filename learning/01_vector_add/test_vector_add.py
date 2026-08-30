"""Correctness tests, especially around block boundaries."""

import pytest
import torch

from vector_add import pytorch_add, triton_add


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.parametrize("size", [1, 127, 128, 129, 256, 1_000, 100_003, 1_000_000])
@pytest.mark.parametrize("block_size", [128, 256, 512, 1024])
def test_matches_pytorch(size: int, block_size: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        triton_add(x, y, block_size), pytorch_add(x, y), rtol=0, atol=0
    )


@requires_cuda
def test_rejects_a_matrix() -> None:
    x = torch.randn((4, 4), device="cuda")
    with pytest.raises(ValueError, match="1D"):
        triton_add(x, x)

