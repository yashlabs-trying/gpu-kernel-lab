import pytest
import torch

from attention import pytorch_attention, triton_attention

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.parametrize("shape", [(1, 1, 7, 16), (1, 2, 33, 64), (2, 4, 65, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_matches_materialized_reference(shape, dtype, causal):
    batch, heads, sequence, dimension = shape
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q); v = torch.randn_like(q)
    lengths = torch.tensor([sequence - index * 3 for index in range(batch)], device="cuda", dtype=torch.int32)
    actual, actual_lse = triton_attention(q, k, v, lengths, lengths, causal)
    expected, expected_lse = pytorch_attention(q, k, v, lengths, lengths, causal)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    valid = torch.arange(sequence, device="cuda")[None, :] < lengths[:, None]
    valid = valid[:, None, :].expand(batch, heads, sequence)
    torch.testing.assert_close(actual_lse[valid], expected_lse[valid], rtol=2e-3, atol=2e-3)

