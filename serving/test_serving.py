import pytest
import torch
from serving.graph_buckets import CUDAGraphBuckets
from serving.paged_kv import PagedKVAllocator
from serving.sampling import Sampler,SamplingParams
from serving.scheduler import ContinuousBatchScheduler,Request,RequestState


def allocator(blocks=8):
    return PagedKVAllocator(num_layers=2,num_blocks=blocks,block_size=4,num_kv_heads=2,head_dim=8,dtype=torch.float32,device='cpu')


def test_paged_allocation_boundary_metadata_and_reuse():
    a=allocator(); a.create('a'); assert a.append_slots('a',5)==[(0,0),(0,1),(0,2),(0,3),(1,0)]
    a.create('b'); a.append_slots('b',1); assert a.blocks('b')==(2,)
    meta=a.metadata(('a','b'),device='cpu')
    assert meta.block_tables.tolist()==[[0,1],[2,-1]] and meta.lengths.tolist()==[5,1]
    assert a.release('a'); a.create('c'); a.append_slots('c',1); assert a.blocks('c')==(0,)


def test_allocation_is_atomic_on_oom():
    a=allocator(1); a.create('a')
    with pytest.raises(MemoryError): a.append_slots('a',5)
    assert a.length('a')==0 and a.free_blocks==1


def test_chunked_prefill_does_not_block_decode():
    a=allocator(32); buckets=CUDAGraphBuckets(batch_sizes=(1,2,4),sequence_buckets=(4,16,64))
    s=ContinuousBatchScheduler(a,buckets,max_batch_size=2,prefill_chunk_size=4,max_prefill_tokens=4)
    s.submit(Request('long',tuple(range(12)),2)); s.submit(Request('short',(1,2),2))
    first=s.plan(); assert len(first.prefill)==1 and len(first.prefill[0].token_ids)==4
    s.complete_prefill('long',4)
    second=s.plan(); assert second.prefill[0].request_id=='short'
    s.complete_prefill('short',2)
    for work in second.prefill[1:]: s.complete_prefill(work.request_id,len(work.token_ids))
    third=s.plan()
    assert tuple(x.request_id for x in third.decode)==('short',) and third.prefill[0].request_id=='long'


def test_finished_request_releases_blocks():
    a=allocator(); s=ContinuousBatchScheduler(a,CUDAGraphBuckets())
    s.submit(Request('x',(1,2),1)); s.plan(); s.complete_prefill('x',2); before=a.free_blocks
    s.plan()
    s.complete_decode('x',3); assert s.requests['x'].state==RequestState.FINISHED
    assert a.free_blocks>before


def test_graph_bucket_rounds_up_both_dimensions():
    b=CUDAGraphBuckets(); assert b.select(3,700)==(4,2048); assert b.select(17,10) is None


def test_sampling_is_deterministic_and_honors_stop():
    params=SamplingParams(temperature=.8,top_k=3,top_p=.9,repetition_penalty=1.1,stop_token_ids=frozenset({2}),seed=7,logprobs=2)
    logits=torch.tensor([0.,1.,3.,2.])
    a=Sampler(params).sample(logits,[2]); b=Sampler(params).sample(logits,[2])
    assert a==b and len(a.top_logprobs)==2
    greedy=Sampler(SamplingParams(stop_token_ids=frozenset({2}))).sample(logits)
    assert greedy.token_id==2 and greedy.stopped
