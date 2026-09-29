import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_fused_paged_gather_matches_reference_exactly():
    from serving.paged_gather import gather_paged_gqa
    torch.manual_seed(11)
    keys=torch.randn((2,7,8,16,128),device='cuda',dtype=torch.bfloat16)
    values=torch.randn_like(keys)
    tables=torch.tensor([[5,1,6],[3,0,-1]],device='cuda',dtype=torch.int32)
    lengths=torch.tensor([37,23],device='cuda',dtype=torch.int32)
    actual_k,actual_v=gather_paged_gqa(keys,values,tables,lengths,1)
    for row,length in enumerate(lengths.tolist()):
        count=(length+15)//16
        expected_k=keys[1,tables[row,:count].long()].permute(1,0,2,3).reshape(8,-1,128)[:,:length].repeat_interleave(2,0)
        expected_v=values[1,tables[row,:count].long()].permute(1,0,2,3).reshape(8,-1,128)[:,:length].repeat_interleave(2,0)
        assert torch.equal(actual_k[row,:,:length],expected_k)
        assert torch.equal(actual_v[row,:,:length],expected_v)
