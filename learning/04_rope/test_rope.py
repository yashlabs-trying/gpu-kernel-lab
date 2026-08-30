import pytest
import torch

from rope import build_rope_tables, pytorch_rope, triton_rope_qk

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 1, 1, 8), (2, 4, 7, 64), (1, 8, 17, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_qk_matches_pytorch(shape, dtype) -> None:
    batch, q_heads, sequence, dimension = shape
    k_heads = max(1, q_heads // 4)  # Exercise grouped-query attention layouts.
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn((batch, k_heads, sequence, dimension), device="cuda", dtype=dtype)
    cosine, sine = build_rope_tables(sequence, dimension, q.device, dtype)
    actual_q, actual_k = triton_rope_qk(q, k, cosine, sine)
    tolerance = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(actual_q, pytorch_rope(q, cosine, sine), **tolerance)
    torch.testing.assert_close(actual_k, pytorch_rope(k, cosine, sine), **tolerance)

