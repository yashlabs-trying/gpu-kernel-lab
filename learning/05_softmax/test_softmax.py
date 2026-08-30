import pytest
import torch

from softmax import pytorch_softmax, triton_softmax

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (8, 127), (4, 128), (2, 1_000), (2, 3_000), (1, 4_096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_matches_pytorch_and_sums_to_one(shape, dtype) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 4
    actual = triton_softmax(x)
    tolerance = {"rtol": 2e-5, "atol": 2e-6} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-3}
    torch.testing.assert_close(actual, pytorch_softmax(x), **tolerance)
    torch.testing.assert_close(
        actual.float().sum(dim=-1), torch.ones(shape[:-1], device="cuda"),
        rtol=2e-3, atol=2e-3,
    )


@requires_cuda
def test_large_logits_remain_finite() -> None:
    x = torch.tensor([[10_000.0, 10_001.0, 9_999.0]], device="cuda")
    output = triton_softmax(x)
    assert torch.isfinite(output).all()
