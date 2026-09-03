import pytest
import torch
from activation_int8 import quantize_int8, dequantize_int8
from cuda_decode import project


def test_calibration_cpu():
    torch.manual_seed(1)
    w, x = torch.randn(13,128), torch.randn(8,128)
    base = quantize_int8(w,x,alphas=(0.,))
    calibrated = quantize_int8(w,x)
    assert calibrated.calibration_mse <= base.calibration_mse
    assert calibrated.data.dtype == torch.int8
    zero = quantize_int8(torch.zeros_like(w),x)
    assert torch.count_nonzero(dequantize_int8(zero)) == 0
    with pytest.raises(ValueError):
        quantize_int8(w[:,:127],x[:,:127])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('warps,persistent', [(1,False),(4,False),(8,False),(4,True)])
@pytest.mark.parametrize('quant', [False,True])
@pytest.mark.parametrize('swiglu', [False,True])
def test_projection(warps,persistent,quant,swiglu):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(256,device='cuda',dtype=torch.bfloat16)
    w = torch.randn(34,256,device='cuda',dtype=torch.bfloat16)
    packed = quantize_int8(w,torch.randn(8,256,device='cuda')) if quant else w
    restored = dequantize_int8(packed) if quant else w.float()
    expected = (restored@x.float()).bfloat16()
    if swiglu:
        a,b = expected.chunk(2)
        expected = torch.nn.functional.silu(a)*b
    actual = project(x,packed,warps=warps,persistent=persistent,swiglu=swiglu)
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_tail_stream_and_reject():
    x = torch.randn(7,device='cuda',dtype=torch.bfloat16)
    w = torch.randn(33,7,device='cuda',dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = project(x,w,warps=1)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(y,(w.float()@x.float()).bfloat16(),rtol=.02,atol=.01)
    with pytest.raises(RuntimeError):
        project(x,w,swiglu=True)
    with pytest.raises(RuntimeError):
        project(x,w,warps=2)
