"""Matched GPU-event benchmark for exact and hybrid Qwen paged decode."""
from __future__ import annotations
import argparse,asyncio,json,statistics,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from serving.paged_kv import PagedKVAllocator
from serving.qwen_executor import QwenPagedExecutor
from serving.scheduler import DecodeWork,PrefillWork


async def benchmark_variant(model,context,steps,warmup,block_size,direct_layers):
    config=model.config; blocks=(context+warmup+steps+block_size-1)//block_size
    allocator=PagedKVAllocator(
        num_layers=len(model.model.layers),num_blocks=blocks,block_size=block_size,
        num_kv_heads=config.num_key_value_heads,head_dim=config.head_dim,
        dtype=torch.bfloat16,device='cuda')
    executor=QwenPagedExecutor(model,allocator,direct_attention_layers=direct_layers)
    request_id='benchmark'; allocator.create(request_id)
    slots=tuple(allocator.append_slots(request_id,context))
    ids=tuple((torch.arange(context)%10000+100).tolist())
    await executor.prefill((PrefillWork(request_id,ids,0,slots),),allocator)
    samples=[]
    for index in range(warmup+steps):
        position=allocator.length(request_id); slot=allocator.append_slots(request_id,1)[0]
        work=(DecodeWork(request_id,100,position,slot),)
        metadata=allocator.metadata((request_id,))
        start,end=torch.cuda.Event(True),torch.cuda.Event(True)
        start.record(); await executor.decode(work,metadata,None); end.record(); end.synchronize()
        if index>=warmup: samples.append(start.elapsed_time(end))
    allocator.release(request_id); executor.release(request_id)
    return {'median_ms':statistics.median(samples),'minimum_ms':min(samples),
            'p99_ms':sorted(samples)[min(len(samples)-1,int(.99*len(samples)))]}


async def run(args,model):
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'warmup':args.warmup,'steps':args.steps,'measurements':[]}
    layers=tuple(args.direct_layers)
    for context in args.contexts:
        exact=await benchmark_variant(model,context,args.steps,args.warmup,args.block_size,())
        hybrid=await benchmark_variant(model,context,args.steps,args.warmup,args.block_size,layers)
        row={'context':context,'direct_layers':layers,'exact':exact,'hybrid':hybrid,
             'speedup':exact['median_ms']/hybrid['median_ms'],
             'latency_reduction_percent':(1-hybrid['median_ms']/exact['median_ms'])*100}
        report['measurements'].append(row); print(json.dumps(row),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B')
    parser.add_argument('--contexts',type=int,nargs='+',default=[128,512,2048,4096])
    parser.add_argument('--direct-layers',type=int,nargs='*',default=[])
    parser.add_argument('--block-size',type=int,default=16)
    parser.add_argument('--warmup',type=int,default=8)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=args.local_files_only).eval().cuda()
    report=asyncio.run(run(args,model))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
