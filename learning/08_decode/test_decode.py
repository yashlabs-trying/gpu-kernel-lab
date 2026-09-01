import pytest
import torch

from decode_gemv import decode_gemv
from fused_qkv_cache import fused_qk_norm_rope_cache

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('k,n',[(1024,2048),(1024,1024),(1024,3072),(3072,1024)])
@pytest.mark.parametrize('split',[1,2,4])
def test_gemv(k,n,split):
    torch.manual_seed(4)
    x=torch.randn(k,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(k,n,device='cuda',dtype=torch.bfloat16)
    torch.testing.assert_close(decode_gemv(x,w,split_k=split),x@w,rtol=.03,atol=.03)


def _norm(x,w):
    fp=x.float()
    return w*(fp*torch.rsqrt(fp.square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)


def _rope(x,cos,sin):
    half=x.shape[-1]//2
    first,second=x[...,:half].float(),x[...,half:].float()
    return torch.cat((first*cos.float()-second*sin.float(),
                      second*cos.float()+first*sin.float()),-1).to(x.dtype)


@pytest.mark.parametrize('batch,position',[(1,0),(1,2047),(3,17)])
def test_qk_norm_rope_static_cache(batch,position):
    torch.manual_seed(5)
    dtype=torch.bfloat16
    q=torch.randn(batch,16,128,device='cuda',dtype=dtype)
    k=torch.randn(batch,8,128,device='cuda',dtype=dtype)
    v=torch.randn_like(k)
    qw=torch.randn(128,device='cuda',dtype=dtype)
    kw=torch.randn(128,device='cuda',dtype=dtype)
    cos=torch.randn(2048,64,device='cuda',dtype=dtype)
    sin=torch.randn_like(cos)
    positions=torch.full((batch,),position,device='cuda',dtype=torch.int32)
    kc=torch.full((batch,8,2048,128),float('nan'),device='cuda',dtype=dtype)
    vc=torch.full_like(kc,float('nan'))
    selected_cos=cos[position][None,None,:]
    selected_sin=sin[position][None,None,:]
    expected_q=_rope(_norm(q,qw),selected_cos,selected_sin)
    expected_k=_rope(_norm(k,kw),selected_cos,selected_sin)
    actual_q=fused_qk_norm_rope_cache(q,k,v,qw,kw,cos,sin,positions,kc,vc)
    torch.testing.assert_close(actual_q,expected_q,rtol=.03,atol=.03)
    for b in range(batch):
        torch.testing.assert_close(kc[b,:,position],expected_k[b],rtol=.03,atol=.03)
        torch.testing.assert_close(vc[b,:,position],v[b],rtol=0,atol=0)
    if position:
        assert torch.isnan(kc[:,:,0]).all()


def test_cache_bounds():
    q=torch.zeros(1,16,128,device='cuda',dtype=torch.bfloat16)
    k=torch.zeros(1,8,128,device='cuda',dtype=torch.bfloat16)
    table=torch.ones(4,64,device='cuda',dtype=torch.bfloat16)
    cache=torch.empty(1,8,4,128,device='cuda',dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='bounds'):
        fused_qk_norm_rope_cache(q,k,k,torch.ones(128,device='cuda',dtype=q.dtype),
            torch.ones(128,device='cuda',dtype=q.dtype),table,table,
            torch.tensor([4],device='cuda',dtype=torch.int32),cache,cache.clone())
