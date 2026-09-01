import pytest
import torch
import torch.nn.functional as F
from fused_projection_swiglu import fused_projection_swiglu

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def reference(x,wg,wu): return F.silu(F.linear(x,wg))*F.linear(x,wu)


@pytest.mark.parametrize('m',[1,13,128,512,2048])
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_fused_projection(m,dtype):
    torch.manual_seed(11)
    x=torch.randn((m,1024),device='cuda',dtype=dtype)
    wg=torch.randn((3072,1024),device='cuda',dtype=dtype)
    wu=torch.randn_like(wg)
    candidate=fused_projection_swiglu(x[0] if m==1 else x,wg,wu)
    expected=reference(x[0] if m==1 else x,wg,wu)
    tolerance=.02 if dtype==torch.float16 else .03
    torch.testing.assert_close(candidate,expected,rtol=tolerance,atol=tolerance)


def test_no_input_padding_in_output():
    x=torch.randn((17,127),device='cuda',dtype=torch.float16)
    wg=torch.randn((129,127),device='cuda',dtype=torch.float16)
    wu=torch.randn_like(wg)
    assert fused_projection_swiglu(x,wg,wu).shape == (17,129)
