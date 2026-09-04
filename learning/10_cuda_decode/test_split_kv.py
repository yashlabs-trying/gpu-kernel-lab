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


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('magnitude',[0.,20.])
def test_stable_softmax_d64(magnitude):
    torch.manual_seed(3)
    q = torch.randn(8,64,device='cuda',dtype=torch.bfloat16)*magnitude
    k = torch.randn(8,257,64,device='cuda',dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scores = (q.float()[:,None,:]*k.float()).sum(-1)/8
    expected = (scores.softmax(-1)[...,None]*v.float()).sum(1).bfloat16()
    actual = split_kv_attention(q,k,v,128)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.002)
    with pytest.raises(ValueError):
        split_kv_attention(q,k[:,:0],v[:,:0])


@pytest.mark.parametrize('position',[0,128,256])
def test_fixed_capacity_valid_position(position):
    torch.manual_seed(19)
    q=torch.randn(16,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(8,272,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    valid=torch.tensor([position],device='cuda',dtype=torch.long)
    local=torch.empty(16,3,128,device='cuda',dtype=torch.float32)
    lse=torch.empty(16,3,device='cuda',dtype=torch.float32)
    output=torch.empty_like(q)
    expected=torch.nn.functional.scaled_dot_product_attention(
        q[None,:,None],k[:,:position+1][None],v[:,:position+1][None],enable_gqa=True)[0,:,0]
    actual=split_kv_attention(q,k,v,128,valid_position=valid,output=output,local=local,lse=lse)
    assert actual.data_ptr()==output.data_ptr()
    torch.testing.assert_close(actual,expected,rtol=.03,atol=.003)
