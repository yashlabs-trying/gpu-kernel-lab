import pytest
import torch

from quantization import PackedInt4, quantize_int4, dequantize_int4


@pytest.mark.parametrize('group', [32, 64, 128])
def test_pack_roundtrip_cpu(group):
    w = torch.arange(-7, 8, dtype=torch.float32).repeat(group)[:group].repeat(3, 1)
    packed = quantize_int4(w, group_size=group, chunk_rows=2)
    torch.testing.assert_close(dequantize_int4(packed), w, rtol=0, atol=0)
    assert packed.storage_bytes == w.numel() // 2 + 3 * 4


def test_zero_groups_and_invalid_input():
    p = quantize_int4(torch.zeros(2, 128))
    assert torch.equal(dequantize_int4(p), torch.zeros(2, 128))
    assert torch.isfinite(p.scales).all()
    with pytest.raises(ValueError):
        quantize_int4(torch.ones(2, 127))
    with pytest.raises(ValueError):
        quantize_int4(torch.full((2, 128), float('nan')))


@pytest.mark.parametrize('group', [32, 64, 128])
def test_quantization_error_bound(group):
    torch.manual_seed(5)
    w = torch.randn(7, 256)
    packed = quantize_int4(w, group)
    bound = packed.scales.repeat_interleave(group, -1) / 2 + 1e-6
    assert ((w - dequantize_int4(packed)).abs() <= bound).all()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@cuda
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('n,k', [(17, 128), (1024, 1024), (2048, 1024), (3072, 1024), (1024, 3072)])
def test_gemv_against_dequantized_oracle(dtype, n, k):
    from kernels import int4_gemv
    torch.manual_seed(42)
    w = torch.randn(n, k, device='cuda', dtype=dtype) / k ** .5
    x = torch.randn(k, device='cuda', dtype=dtype)
    packed = quantize_int4(w)
    expected = (dequantize_int4(packed) @ x.float()).to(dtype)
    actual = int4_gemv(x, packed)
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)


@cuda
@pytest.mark.parametrize('n', [17, 65, 151936])
@pytest.mark.parametrize('top', [1, 4, 8])
def test_fused_head_selection(n, top):
    from kernels import int4_gemv, int4_lm_head_topk
    torch.manual_seed(13)
    w = torch.randn(n, 128, device='cuda', dtype=torch.bfloat16) / 128 ** .5
    x = torch.randn(128, device='cuda', dtype=w.dtype)
    packed = quantize_int4(w)
    logits = int4_gemv(x, packed).float()
    scores, ids = int4_lm_head_topk(x, packed, top_k=top)
    torch.testing.assert_close(scores, logits.topk(top).values, rtol=0, atol=0)
    torch.testing.assert_close(scores, logits[ids], rtol=0, atol=0)
    assert ids.unique().numel() == top
    if top == 1:
        assert ids.item() == logits.argmax().item()


@cuda
def test_argmax_tie():
    from kernels import int4_lm_head_topk
    p = quantize_int4(torch.zeros(65, 128, device='cuda', dtype=torch.bfloat16))
    _, ids = int4_lm_head_topk(torch.ones(128, device='cuda', dtype=torch.bfloat16), p)
    assert ids.item() == 0
