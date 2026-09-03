import pytest
import torch
from split_kv import split_kv_attention


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('length',[1,7,129,2048])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('tile',[64,128,256])
def test_split(length,dtype,tile):
    torch.manual_seed(7)
    q = torch.randn(16,128,device='cuda',dtype=dtype)
    # Noncontiguous head/sequence layout like a cache view.
    k = torch.randn(length,8,128,device='cuda',dtype=dtype).transpose(0,1)
    v = torch.randn_like(k)
    kr,vr = k.float().repeat_interleave(2,0),v.float().repeat_interleave(2,0)
    scores = (q.float()[:,None,:]*kr).sum(-1)/(128**.5)
    expected = (scores.softmax(-1)[...,None]*vr).sum(1).to(dtype)
    actual = split_kv_attention(q,k,v,tile)
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.002)
