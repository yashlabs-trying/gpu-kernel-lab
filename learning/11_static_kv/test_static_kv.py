import pytest
import torch
from static_kv import advance_position,make_mask,open_mask


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_persistent_mask_and_position():
    mask=make_mask(1,11,3,torch.bfloat16,'cuda')
    position=torch.tensor([3],device='cuda',dtype=torch.long)
    mask_pointer,position_pointer=mask.data_ptr(),position.data_ptr()
    open_mask(mask,position)
    advance_position(position)
    torch.cuda.synchronize()
    assert position.item()==4
    assert torch.all(mask[..., :4]==0)
    assert torch.all(mask[..., 4:]<0)
    assert mask.data_ptr()==mask_pointer and position.data_ptr()==position_pointer


def test_mask_bounds():
    with pytest.raises(ValueError):
        make_mask(1,4,5,torch.bfloat16,'cuda')
