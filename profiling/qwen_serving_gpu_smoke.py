"""End-to-end GPU smoke for scheduler -> Qwen -> paged KV -> sampling."""
from __future__ import annotations
import argparse,asyncio,json,math,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from profiling.accuracy_utils import deterministic_prompts
from serving.engine import ServingEngine
from serving.graph_buckets import CUDAGraphBuckets
from serving.paged_kv import PagedKVAllocator
from serving.qwen_executor import QwenPagedExecutor
from serving.sampling import SamplingParams
from serving.scheduler import ContinuousBatchScheduler,Request


async def run(args,model,tokenizer):
    prompts=deterministic_prompts(args.requests,args.seed)
    encoded=[tokenizer(x,add_special_tokens=True)['input_ids'][:args.max_input_tokens] for x in prompts]
    expected=[]
    with torch.inference_mode():
        for tokens in encoded:
            ids=torch.tensor([tokens],device='cuda',dtype=torch.long)
            mask=torch.ones_like(ids)
            output=model(input_ids=ids,attention_mask=mask,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values; generated=[]
            for step in range(args.max_new_tokens):
                token=int(output.logits[0,-1].argmax()); generated.append(token)
                if step+1<args.max_new_tokens:
                    next_token=torch.tensor([[token]],device='cuda',dtype=torch.long)
                    mask=torch.cat((mask,torch.ones_like(next_token)),dim=1)
                    output=model(input_ids=next_token,attention_mask=mask,
                                 past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache=output.past_key_values
            expected.append(generated)
    blocks_per_request=math.ceil((args.max_input_tokens+args.max_new_tokens)/args.block_size)
    config=model.config
    allocator=PagedKVAllocator(
        num_layers=len(model.model.layers),num_blocks=blocks_per_request*args.requests,
        block_size=args.block_size,num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,dtype=torch.bfloat16,device='cuda')
    scheduler=ContinuousBatchScheduler(
        allocator,CUDAGraphBuckets(),max_batch_size=args.requests,
        prefill_chunk_size=args.prefill_chunk,max_prefill_tokens=args.prefill_chunk*args.requests)
    engine=ServingEngine(scheduler,QwenPagedExecutor(model,allocator),idle_sleep_s=0)
    queues=[]
    for index,tokens in enumerate(encoded):
        queues.append(await engine.submit(Request(
            f'gpu-{index}',tuple(tokens),args.max_new_tokens,
            sampling=SamplingParams(temperature=0),timeout_s=120)))
    while engine.active_requests:
        await engine.step()
    events=[]
    for queue in queues:
        request_events=[]
        while not queue.empty():
            event=queue.get_nowait()
            request_events.append({'token_id':event.token_id,'finished':event.finished,
                                   'finish_reason':event.finish_reason,'error':event.error})
        events.append(request_events)
    snapshot=engine.snapshot()
    actual=[[x['token_id'] for x in rows if x['token_id'] is not None] for rows in events]
    baseline_match=actual==expected
    passed=(snapshot['requests']['completed']==args.requests and
            snapshot['requests']['failed']==0 and allocator.free_blocks==allocator.num_blocks and
            baseline_match and
            all(sum(x['token_id'] is not None for x in rows)==args.max_new_tokens for rows in events))
    return {'pass':passed,'requests':args.requests,'max_new_tokens':args.max_new_tokens,
            'metrics':snapshot,'all_blocks_reclaimed':allocator.free_blocks==allocator.num_blocks,
            'baseline_token_match':baseline_match,'expected_tokens':expected,
            'actual_tokens':actual,'events':events}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B')
    parser.add_argument('--requests',type=int,default=4)
    parser.add_argument('--max-new-tokens',type=int,default=4)
    parser.add_argument('--max-input-tokens',type=int,default=128)
    parser.add_argument('--prefill-chunk',type=int,default=64)
    parser.add_argument('--block-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20260930)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=args.local_files_only)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,dtype=torch.bfloat16,attn_implementation='sdpa',
        local_files_only=args.local_files_only).eval().cuda()
    report=asyncio.run(run(args,model,tokenizer))
    report['environment']={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['pass']: raise SystemExit(2)


if __name__=='__main__': main()
