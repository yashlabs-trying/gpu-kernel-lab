"""Matched DynamicCache, StaticCache and fused direct-write Qwen decode."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/11_static_kv'))
from static_kv import advance_position,install_fused_static_kv,make_mask,new_static_cache,open_mask
from model_ablation import install


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[128,512,2048,4096])
    p.add_argument('--steps',type=int,default=16)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--mode',choices=['benchmark','capture'],default='benchmark')
    p.add_argument('--capture-variant',choices=['dynamic','static','fused'],default='fused')
    p.add_argument('--hybrid',action='store_true',help='compose the prior QKV/MLP projection fusion')
    p.add_argument('--split-attention',action='store_true',help='replace masked SDPA with preallocated split-KV GQA')
    args=p.parse_args()
    if args.mode=='capture' and len(args.lengths)!=1:
        raise ValueError('capture takes exactly one context length')
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
    install(model,'qwen_rms_cuda_hybrid' if args.hybrid else 'qwen_rms')
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'dtype':'bfloat16',
        'backend':model.config._attn_implementation,'hybrid':args.hybrid,'split_attention':args.split_attention,'steps':args.steps,'warmup':args.warmup,
        'repeats':args.repeats,'method':'B1 cache-enabled greedy decode; argmax included; cache setup/prefill outside timed region','measurements':[]}
    capacity=max(args.lengths)+args.steps
    report['capacity']=capacity

    def prepare_dynamic(length):
        ids=(torch.arange(length,device='cuda')[None,:]%10000+100).long()
        mask=torch.ones_like(ids)
        out=model(input_ids=ids,attention_mask=mask,use_cache=True,logits_to_keep=1)
        return {'cache':out.past_key_values,'token':out.logits[:,-1:].argmax(-1),'mask':mask}

    def dynamic(state):
        state['mask']=torch.cat((state['mask'],torch.ones((1,1),device='cuda',dtype=torch.long)),1)
        out=model(input_ids=state['token'],attention_mask=state['mask'],past_key_values=state['cache'],use_cache=True,logits_to_keep=1)
        state['cache'],state['token']=out.past_key_values,out.logits[:,-1:].argmax(-1)

    def prepare_static(length):
        ids=(torch.arange(length,device='cuda')[None,:]%10000+100).long()
        cache=new_static_cache(model,capacity)
        out=model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True,logits_to_keep=1)
        position=torch.tensor([length],device='cuda',dtype=torch.long)
        fixed=make_mask(1,capacity,length,torch.bfloat16,'cuda')
        return {'cache':cache,'token':out.logits[:,-1:].argmax(-1),'position':position,
            'position_ids':position.view(1,1),'mask':fixed,'mapping':{'full_attention':fixed}}

    def static(state,fused):
        if not fused:
            open_mask(state['mask'],state['position'])
        out=model(input_ids=state['token'],attention_mask=state['mapping'],position_ids=state['position_ids'],
            past_key_values=state['cache'],use_cache=True,logits_to_keep=1)
        state['token']=out.logits[:,-1:].argmax(-1)
        advance_position(state['position'])

    def measure(name,length,prepare,step):
        for _ in range(args.warmup):
            state=prepare(length)
            for _ in range(args.steps): step(state)
        samples=[]
        count=1 if args.mode=='capture' else args.repeats
        for _ in range(count):
            state=prepare(length)
            torch.cuda.synchronize()
            if args.mode=='capture': torch.cuda.cudart().cudaProfilerStart()
            begin,end=torch.cuda.Event(True),torch.cuda.Event(True)
            begin.record(); wall=time.perf_counter()
            for _ in range(args.steps): step(state)
            end.record(); end.synchronize()
            if args.mode=='capture': torch.cuda.cudart().cudaProfilerStop()
            samples.append(begin.elapsed_time(end)/args.steps)
        row={'variant':name,'context':length,'gpu_samples_ms_per_token':samples,
            'gpu_median_ms_per_token':statistics.median(samples)}
        report['measurements'].append(row); print(json.dumps(row),flush=True)

    for length in args.lengths:
        if args.mode=='benchmark' or args.capture_variant=='dynamic':
            measure('dynamic',length,prepare_dynamic,dynamic)
        if args.mode=='benchmark' or args.capture_variant=='static':
            measure('static',length,prepare_static,lambda state:static(state,False))
    install_fused_static_kv(model,capacity,split_attention=args.split_attention)
    for length in args.lengths:
        if args.mode=='benchmark' or args.capture_variant=='fused':
            def prepare_fused(length):
                return prepare_static(length)
            measure('fused',length,prepare_fused,lambda state:static(state,True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
