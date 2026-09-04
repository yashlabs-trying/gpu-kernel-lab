"""Long control-plane stress test; intentionally uses a deterministic fake executor."""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from serving.engine import ServingEngine
from serving.graph_buckets import CUDAGraphBuckets
from serving.paged_kv import PagedKVAllocator
from serving.scheduler import ContinuousBatchScheduler,Request


class Executor:
    async def prefill(self,work,allocator): await asyncio.sleep(0)
    async def decode(self,work,metadata,graph_batch_size):
        await asyncio.sleep(0)
        return {x.request_id:torch.arange(32,dtype=torch.float32) for x in work}


async def run(count,seed):
    random.seed(seed)
    allocator=PagedKVAllocator(num_layers=1,num_blocks=max(1024,count*12),block_size=16,
        num_kv_heads=1,head_dim=1,dtype=torch.float32,device='cpu')
    scheduler=ContinuousBatchScheduler(allocator,CUDAGraphBuckets(),max_batch_size=16,
        prefill_chunk_size=64,max_prefill_tokens=512)
    engine=ServingEngine(scheduler,Executor(),idle_sleep_s=0)
    queues=[]; started=time.perf_counter()
    for index in range(count):
        prompt=tuple(range(random.randint(1,256))); maximum=random.randint(1,16)
        request=Request(f'stress-{index}',prompt,maximum,priority=random.randint(-2,2),timeout_s=30)
        queues.append(await engine.submit(request))
    cancelled=0
    for index in range(0,count,37): cancelled+=bool(await engine.cancel(f'stress-{index}'))
    while engine.active_requests: await engine.step()
    elapsed=time.perf_counter()-started; snapshot=engine.snapshot()
    result={'requests':count,'seed':seed,'wall_s':elapsed,'requests_per_second':count/elapsed,
        'generated_tokens_per_second_wall':snapshot['tokens']['generated']/elapsed,
        'cancelled_by_test':cancelled,'metrics':snapshot,
        'all_blocks_reclaimed':allocator.free_blocks==allocator.num_blocks,
        'terminal_events_present':all(not queue.empty() for queue in queues)}
    if not result['all_blocks_reclaimed'] or not result['terminal_events_present'] or snapshot['requests']['failed']:
        raise RuntimeError(result)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--requests',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20260904); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=asyncio.run(run(args.requests,args.seed))
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
