import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_direct_paged_attention_matches_gathered_sdpa():
    from serving.paged_attention import paged_attention
    torch.manual_seed(7)
    batch,heads,kv_heads,dimension=2,16,8,128
    block_size,blocks=16,6
    keys=torch.randn((1,blocks,kv_heads,block_size,dimension),device='cuda',dtype=torch.bfloat16)
    values=torch.randn_like(keys); q=torch.randn((batch,heads,dimension),device='cuda',dtype=torch.bfloat16)
    tables=torch.tensor([[4,1,5],[2,0,-1]],device='cuda',dtype=torch.int32)
    lengths=torch.tensor([37,23],device='cuda',dtype=torch.int32)
    actual=paged_attention(q,keys,values,tables,lengths,0,block_size)
    expected=[]
    for row,length in enumerate(lengths.tolist()):
        count=(length+block_size-1)//block_size
        k=keys[0,tables[row,:count].long()].permute(1,0,2,3).reshape(kv_heads,-1,dimension)[:,:length]
        v=values[0,tables[row,:count].long()].permute(1,0,2,3).reshape(kv_heads,-1,dimension)[:,:length]
        k=k.repeat_interleave(heads//kv_heads,0); v=v.repeat_interleave(heads//kv_heads,0)
        expected.append(torch.nn.functional.scaled_dot_product_attention(q[row,:,None],k,v)[:,0])
    torch.testing.assert_close(actual,torch.stack(expected),atol=.02,rtol=.02)
