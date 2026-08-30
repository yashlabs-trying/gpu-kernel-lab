"""Compare materialized reference attention with tiled online attention."""
import argparse, csv, statistics
from pathlib import Path
import torch
from attention import pytorch_attention, triton_attention


def timing(fn, warmup=5, repeats=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); samples=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); samples.append(a.elapsed_time(b))
    return statistics.median(samples)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shapes",nargs="+",default=["1x8x128x64","1x8x512x64","1x8x1024x64"])
    p.add_argument("--dtype",choices=["float16","bfloat16"],default="bfloat16")
    p.add_argument("--output",type=Path); args=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    dtype=getattr(torch,args.dtype); results=[]
    configs=[(16,32,4,2),(32,32,4,3),(64,32,4,3),(64,64,8,4)]
    for text in args.shapes:
        batch,heads,sequence,dimension=map(int,text.split("x"))
        q=torch.randn((batch,heads,sequence,dimension),device="cuda",dtype=dtype)
        k=torch.randn_like(q); v=torch.randn_like(q)
        lengths=torch.full((batch,),sequence,device="cuda",dtype=torch.int32)
        # Two FP32 SxS tensors approximate materialized scores + probabilities.
        matrix_mib=2*batch*heads*sequence*sequence*4/(1024**2)
        ms=timing(lambda:pytorch_attention(q,k,v,lengths,lengths,True))
        results.append(["materialized_reference",text,"framework","framework","framework","framework",ms,matrix_mib])
        for bm,bn,warps,stages in configs:
            ms=timing(lambda bm=bm,bn=bn,w=warps,s=stages:triton_attention(q,k,v,lengths,lengths,True,bm,bn,w,s))
            results.append(["tiled_online",text,bm,bn,warps,stages,ms,matrix_mib])
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {args.dtype}")
    for row in results: print(row)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["implementation","BxHxSxD","BLOCK_M","BLOCK_N","warps","stages","median_ms","avoided_score_probability_MiB"]); w.writerows(results)


if __name__=="__main__": main()

