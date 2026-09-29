"""Direct block-table attention versus gather-plus-SDPA decode baseline."""
from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from serving.paged_attention import paged_attention
from serving.qwen_executor import materialize_paged_kv


def measure(call,warmup,repeats):
    for _ in range(warmup): call()
    torch.cuda.synchronize(); samples=[]
    for _ in range(repeats):
        start,end=torch.cuda.Event(True),torch.cuda.Event(True)
        start.record(); call(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end)*1000)
    return {'median_us':statistics.median(samples),'minimum_us':min(samples),
            'p99_us':sorted(samples)[min(len(samples)-1,int(.99*len(samples)))]}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lengths',type=int,nargs='+',default=[128,512,2048,4096])
    parser.add_argument('--block-size',type=int,default=16)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); torch.manual_seed(7)
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'block_size':args.block_size,
            'warmup':args.warmup,'repeats':args.repeats,'measurements':[]}
    for length in args.lengths:
        blocks=(length+args.block_size-1)//args.block_size
        q=torch.randn((1,16,128),device='cuda',dtype=torch.bfloat16)
        keys=torch.randn((1,blocks,8,args.block_size,128),device='cuda',dtype=torch.bfloat16)
        values=torch.randn_like(keys)
        table=torch.arange(blocks-1,-1,-1,device='cuda',dtype=torch.int32)[None]
        lengths=torch.tensor([length],device='cuda',dtype=torch.int32)
        def gathered():
            k=materialize_paged_kv(keys,0,table[0],length,args.block_size)[None].repeat_interleave(2,1)
            v=materialize_paged_kv(values,0,table[0],length,args.block_size)[None].repeat_interleave(2,1)
            return torch.nn.functional.scaled_dot_product_attention(q[:,:,None],k,v)[:,:,0]
        expected=gathered(); actual=paged_attention(q,keys,values,table,lengths,0,args.block_size)
        error=float((actual.float()-expected.float()).abs().max())
        row={'context':length,'max_abs':error,'gather_sdpa':measure(gathered,args.warmup,args.repeats),
             'direct_paged':measure(lambda:paged_attention(q,keys,values,table,lengths,0,args.block_size),
                                    args.warmup,args.repeats)}
        row['speedup']=row['gather_sdpa']['median_us']/row['direct_paged']['median_us']
        report['measurements'].append(row); print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
