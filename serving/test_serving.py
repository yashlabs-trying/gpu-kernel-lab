import pytest
import torch
import asyncio
import time
from serving.engine import ServingEngine
from serving.graph_buckets import CUDAGraphBuckets
from serving.paged_kv import PagedKVAllocator
from serving.sampling import Sampler,SamplingParams
from serving.scheduler import AdmissionError,ContinuousBatchScheduler,Request,RequestState
from serving.api import create_app
from fastapi.testclient import TestClient


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


def test_admission_timeout_priority_and_reclamation():
    a=allocator(4); s=ContinuousBatchScheduler(a,CUDAGraphBuckets(),max_batch_size=2,prefill_chunk_size=2)
    s.submit(Request('low',(1,2),2,priority=0)); s.submit(Request('high',(3,4),2,priority=10))
    with pytest.raises(AdmissionError): s.submit(Request('too-large',tuple(range(20)),1))
    plan=s.plan(); assert plan.prefill[0].request_id=='high'
    for work in plan.prefill: s.complete_prefill(work.request_id,len(work.token_ids))
    s.requests['low'].submitted_at=time.monotonic()-100
    assert s.expire()==('low',) and s.requests['low'].state==RequestState.TIMED_OUT


class FakeExecutor:
    async def prefill(self,work,allocator): await asyncio.sleep(0)
    async def decode(self,work,metadata,graph_batch_size):
        await asyncio.sleep(0)
        return {x.request_id:torch.tensor([0.,1.,3.,2.]) for x in work}


def test_concurrent_stress_and_metrics():
    async def run():
        a=allocator(1024); scheduler=ContinuousBatchScheduler(a,CUDAGraphBuckets(),max_batch_size=8,prefill_chunk_size=8,max_prefill_tokens=32)
        engine=ServingEngine(scheduler,FakeExecutor())
        requests=[Request(f'r{i}',tuple(range(1+(i%31))),1+(i%3),timeout_s=10) for i in range(100)]
        queues=await asyncio.gather(*(engine.submit(r) for r in requests))
        while engine.active_requests: assert await engine.step()
        snapshot=engine.snapshot()
        assert snapshot['requests']['completed']==100 and snapshot['requests']['failed']==0
        assert snapshot['tokens']['generated']==sum(r.max_new_tokens for r in requests)
        assert a.free_blocks==a.num_blocks
        assert all(not q.empty() for q in queues)
    asyncio.run(run())


class FakeTokenizer:
    def encode(self,text,add_special_tokens=True): return [1]+[2+(ord(c)%13) for c in text]
    def decode(self,ids,skip_special_tokens=False): return ''.join(chr(97+(i%26)) for i in ids)
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return '\n'.join(f"{x['role']}: {x['content']}" for x in messages)+'\nassistant:'


def api_client(blocks=64):
    a=allocator(blocks); scheduler=ContinuousBatchScheduler(a,CUDAGraphBuckets(),max_batch_size=4,prefill_chunk_size=8)
    return TestClient(create_app(ServingEngine(scheduler,FakeExecutor()),FakeTokenizer()))


def test_openai_json_streaming_health_and_metrics():
    with api_client() as client:
        assert client.get('/health').json()['status']=='ok'
        assert client.get('/v1/models').status_code==200
        response=client.post('/v1/completions',json={'prompt':'hello','max_tokens':2,'seed':4})
        assert response.status_code==200 and response.json()['usage']=={'prompt_tokens':6,'completion_tokens':2,'total_tokens':8}
        with client.stream('POST','/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}],'stream':True,'max_tokens':2}) as stream:
            body=''.join(stream.iter_text())
        assert stream.status_code==200 and 'chat.completion.chunk' in body and 'data: [DONE]' in body
        metrics=client.get('/metrics').json(); assert metrics['requests']['completed']==2
        assert metrics['tokens']['generated']==4 and metrics['kv']['used_blocks']==0


def test_api_validation_unknown_model_and_capacity_rejection():
    with api_client(1) as client:
        assert client.post('/v1/completions',json={'prompt':'x','max_tokens':0}).status_code==422
        assert client.post('/v1/completions',json={'model':'missing','prompt':'x'}).status_code==404
        rejected=client.post('/v1/completions',json={'prompt':'long','max_tokens':8})
        assert rejected.status_code==503 and rejected.headers['retry-after']=='1'
