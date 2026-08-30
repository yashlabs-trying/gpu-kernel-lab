import pytest
import torch

from gemm import pytorch_gemm, triton_gemm

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 13, 9), (64, 64, 64), (127, 129, 65), (256, 256, 256), (512, 1024, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matches_pytorch(shape, dtype) -> None:
    m_size, n_size, k_size = shape
    torch.manual_seed(0)
    a = torch.randn((m_size, k_size), device="cuda", dtype=dtype)
    b = torch.randn((k_size, n_size), device="cuda", dtype=dtype)
    actual = triton_gemm(a, b)
    expected = pytorch_gemm(a, b)
    tolerance = {"rtol": 2e-2, "atol": 2e-2} if dtype == torch.float16 else {"rtol": 3e-2, "atol": 3e-2}
    torch.testing.assert_close(actual, expected, **tolerance)


@requires_cuda
@pytest.mark.parametrize("config", [(32, 32, 32, 4, 2), (64, 64, 32, 4, 3), (128, 64, 32, 8, 4), (64, 128, 64, 8, 4)])
def test_tuning_configurations(config) -> None:
    bm, bn, bk, warps, stages = config
    a = torch.randn((129, 131), device="cuda", dtype=torch.float16)
    b = torch.randn((131, 127), device="cuda", dtype=torch.float16)
    actual = triton_gemm(a, b, block_m=bm, block_n=bn, block_k=bk, num_warps=warps, num_stages=stages)
    torch.testing.assert_close(actual, a @ b, rtol=2e-2, atol=2e-2)

