"""Concurrent real-GPU load test for the production Qwen paged executor."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.engine import ServingEngine
from serving.graph_buckets import CUDAGraphBuckets
from serving.kernel_policy import load_qwen_policy
from serving.paged_kv import PagedKVAllocator
from serving.qwen_executor import QwenPagedExecutor
from serving.scheduler import ContinuousBatchScheduler,Request


def build_runtime(model,policy,args):
    required=sum(math.ceil((length+args.output_tokens)/policy.block_size)
                 for length in args.prompt_lengths)
    blocks=max(required*math.ceil(args.requests/len(args.prompt_lengths)),256)
    allocator=PagedKVAllocator(
        num_layers=len(model.model.layers),num_blocks=blocks,
        block_size=policy.block_size,num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,dtype=torch.bfloat16,device='cuda')
    scheduler=ContinuousBatchScheduler(
        allocator,CUDAGraphBuckets(batch_sizes=(1,2,4,8,16)),
        max_batch_size=args.max_batch_size,prefill_chunk_size=args.prefill_chunk,
        max_prefill_tokens=args.prefill_budget)
    return allocator,scheduler,QwenPagedExecutor(model,allocator,policy=policy)


async def execute(engine,requests):
    queues=await asyncio.gather(*(engine.submit(request) for request in requests))
    while engine.active_requests:
        if not await engine.step(): await asyncio.sleep(0)
    # Ensure every request emitted a terminal event.
    for queue in queues:
        events=[]
        while not queue.empty(): events.append(queue.get_nowait())
        if not events or not events[-1].finished:
            raise RuntimeError('request did not emit a terminal event')


def requests_for(args,prefix):
    result=[]
    for index in range(args.requests):
        length=args.prompt_lengths[index%len(args.prompt_lengths)]
        # Deterministic valid non-special IDs; tokenization is outside this GPU test.
        tokens=tuple(100+((index*131+position*17)%30000) for position in range(length))
        result.append(Request(f'{prefix}-{index}',tokens,args.output_tokens,timeout_s=600))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B')
    parser.add_argument('--requests',type=int,default=16)
    parser.add_argument('--prompt-lengths',type=int,nargs='+',default=[32,64,128,256])
    parser.add_argument('--output-tokens',type=int,default=16)
    parser.add_argument('--max-batch-size',type=int,default=8)
    parser.add_argument('--prefill-chunk',type=int,default=128)
    parser.add_argument('--prefill-budget',type=int,default=512)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.requests,args.output_tokens,args.max_batch_size,args.prefill_chunk,
           args.prefill_budget,*args.prompt_lengths)<=0:
        raise SystemExit('all workload dimensions must be positive')
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=args.local_files_only).eval().cuda()
    policy=load_qwen_policy(model)

    # Warm kernels and allocator paths without contaminating measured percentiles.
    allocator,scheduler,executor=build_runtime(model,policy,args)
    warm_args=argparse.Namespace(**vars(args)); warm_args.requests=1
    asyncio.run(execute(ServingEngine(scheduler,executor),requests_for(warm_args,'warm')))
    del allocator,scheduler,executor
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    allocator,scheduler,executor=build_runtime(model,policy,args)
    engine=ServingEngine(scheduler,executor)
    started=time.perf_counter()
    asyncio.run(execute(engine,requests_for(args,'load')))
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    report={
        'schema_version':1,
        'environment':{
            'gpu':torch.cuda.get_device_name(),
            'compute_capability':list(torch.cuda.get_device_capability()),
            'torch':torch.__version__,
        },
        'policy':{
            'name':policy.profile,'architecture':policy.architecture,
            'calibrated':policy.calibrated,'reason':policy.reason,
        },
        'workload':{
            'requests':args.requests,'prompt_lengths':args.prompt_lengths,
            'output_tokens':args.output_tokens,'max_batch_size':args.max_batch_size,
            'prefill_chunk':args.prefill_chunk,'prefill_budget':args.prefill_budget,
            'tokenization_and_http_excluded':True,
        },
        'wall_time_s':elapsed,
        'completed_requests_per_second':args.requests/elapsed,
        'metrics':engine.snapshot(),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    checks=(
        report['metrics']['requests']['completed']==args.requests,
        report['metrics']['requests']['failed']==0,
        report['metrics']['kv']['used_blocks']==0,
    )
    raise SystemExit(0 if all(checks) else 1)


if __name__=='__main__': main()
