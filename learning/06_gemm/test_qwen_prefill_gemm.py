import pytest
import torch
from qwen_prefill_gemm import qwen_prefill_gemm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('m', [13, 16, 17, 128, 512, 2048, 4096])
@pytest.mark.parametrize('n,k', [(2048, 1024), (1024, 1024), (3072, 1024), (1024, 3072)])
def test_qwen_shapes(m, n, k):
    torch.manual_seed(1)
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    # Use an explicit FP32 oracle. Default cuBLAS may enable BF16 reduced-
    # precision reduction, which is a different numerical policy.
    expected = (a.float() @ b.float()).to(dtype)
    torch.testing.assert_close(qwen_prefill_gemm(a, b), expected, rtol=3e-2, atol=3e-2)
