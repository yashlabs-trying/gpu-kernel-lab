"""Acceptance test for the real Qwen paged-cache serving executor."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from profiling.accuracy_utils import StreamingTensorMetrics,deterministic_prompts
from serving.paged_kv import PagedKVAllocator
from serving.qwen_executor import QwenPagedExecutor
from serving.scheduler import DecodeWork,PrefillWork


async def paged_prefill(executor,allocator,request_id,token_ids):
    allocator.create(request_id)
    slots=tuple(allocator.append_slots(request_id,len(token_ids)))
    work=PrefillWork(request_id,tuple(token_ids),0,slots)
    return (await executor.prefill((work,),allocator))[request_id]


async def paged_decode(executor,allocator,request_id,token_id,position):
    slot=allocator.append_slots(request_id,1)[0]
    work=DecodeWork(request_id,int(token_id),position,slot)
    metadata=allocator.metadata((request_id,))
    return (await executor.decode((work,),metadata,None))[request_id]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B')
    parser.add_argument('--prompts',type=int,default=64)
    parser.add_argument('--decode-steps',type=int,default=4)
    parser.add_argument('--max-input-tokens',type=int,default=128)
    parser.add_argument('--block-size',type=int,default=16)
    parser.add_argument('--direct-attention',action='store_true')
    parser.add_argument('--direct-layers',type=int,nargs='*',default=None,
                        help='use direct paged attention only in these decoder layers')
    parser.add_argument('--native-gqa',action='store_true',help='gather Hkv heads and use SDPA native GQA')
    parser.add_argument('--fused-projections',action='store_true')
    parser.add_argument('--fused-qkv',action='store_true')
    parser.add_argument('--fused-mlp',action='store_true')
    parser.add_argument('--seed',type=int,default=20260930)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--no-fail',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.prompts,args.decode_steps,args.max_input_tokens,args.block_size)<=0:
        raise ValueError('workload sizes must be positive')
    if args.block_size&(args.block_size-1):
        raise ValueError('block size must be a power of two')

    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=args.local_files_only)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=args.local_files_only).eval().cuda()
    config=model.config
    # Requests are evaluated one at a time and immediately reclaimed.
    capacity=math.ceil((args.max_input_tokens+args.decode_steps)/args.block_size)
    allocator=PagedKVAllocator(
        num_layers=len(model.model.layers),num_blocks=capacity,
        block_size=args.block_size,num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,dtype=torch.bfloat16,device='cuda')
    if args.direct_attention and args.direct_layers is not None:
        raise ValueError('choose --direct-attention or --direct-layers, not both')
    executor=QwenPagedExecutor(model,allocator,direct_attention=args.direct_attention,
                               direct_attention_layers=args.direct_layers,native_gqa=args.native_gqa)
    if args.fused_projections or args.fused_qkv or args.fused_mlp:
        executor.enable_fused_projections(
            qkv=args.fused_projections or args.fused_qkv,
            mlp=args.fused_projections or args.fused_mlp)
    metrics=StreamingTensorMetrics(); prefill_matches=decode_matches=0

    for index,prompt in enumerate(deterministic_prompts(args.prompts,args.seed)):
        encoded=tokenizer(prompt,return_tensors='pt',truncation=True,
                          max_length=args.max_input_tokens).to('cuda')
        token_ids=encoded.input_ids[0].tolist(); request_id=f'quality-{index}'
        baseline=model(**encoded,use_cache=True,logits_to_keep=1)
        baseline_logits=baseline.logits[0,-1].float(); cache=baseline.past_key_values
        paged_logits=asyncio.run(paged_prefill(executor,allocator,request_id,token_ids))
        metrics.update(paged_logits,baseline_logits)
        prefill_matches+=int(paged_logits.argmax()==baseline_logits.argmax())
        mask=encoded.attention_mask

        for step in range(args.decode_steps):
            token=baseline_logits.argmax().view(1,1)
            mask=torch.cat((mask,torch.ones_like(token)),dim=1)
            baseline=model(input_ids=token,attention_mask=mask,past_key_values=cache,
                           use_cache=True,logits_to_keep=1)
            baseline_logits=baseline.logits[0,-1].float(); cache=baseline.past_key_values
            paged_logits=asyncio.run(paged_decode(
                executor,allocator,request_id,int(token.item()),len(token_ids)+step))
            metrics.update(paged_logits,baseline_logits)
            decode_matches+=int(paged_logits.argmax()==baseline_logits.argmax())
        allocator.release(request_id)

    values=metrics.as_dict()
    prefill_agreement=prefill_matches/args.prompts
    decode_count=args.prompts*args.decode_steps
    decode_agreement=decode_matches/decode_count
    checks={
        'prefill_argmax':prefill_agreement>=.99,
        'decode_argmax':decode_agreement>=.99,
        'logit_cosine':values['min_cosine']>=.999,
        'finite':values['finite'],
        'cache_reclaimed':allocator.free_blocks==allocator.num_blocks,
    }
    report={
        'schema_version':1,'model':args.model,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,
                       'gpu':torch.cuda.get_device_name(),'compute_capability':torch.cuda.get_device_capability()},
        'workload':{'prompts':args.prompts,'decode_steps':args.decode_steps,
                    'max_input_tokens':args.max_input_tokens,'block_size':args.block_size,
                    'direct_attention':args.direct_attention,
                    'direct_layers':args.direct_layers,'native_gqa':args.native_gqa,
                    'fused_qkv':args.fused_projections or args.fused_qkv,
                    'fused_mlp':args.fused_projections or args.fused_mlp,'seed':args.seed},
        'prefill_argmax_agreement':prefill_agreement,
        'decode_argmax_agreement':decode_agreement,
        'logits':values,'gate':{'pass':all(checks.values()),'checks':checks},
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['gate']['pass'] and not args.no_fail:
        raise SystemExit(2)


if __name__=='__main__': main()
