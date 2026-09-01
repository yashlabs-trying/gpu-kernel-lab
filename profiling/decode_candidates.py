"""Hot microbenchmarks for decode GEMV and fused Q/K/cache candidates."""
import argparse,json,sys
from pathlib import Path
import torch,triton

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'learning/08_decode')]
from decode_gemv import decode_gemv
from fused_qkv_cache import fused_qk_norm_rope_cache


def bench(fn): return float(triton.testing.do_bench(fn,warmup=100,rep=300))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results/decode_candidates.json'); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU required')
    torch.manual_seed(9); out={'gpu':torch.cuda.get_device_name(),'gemv':[],'qk_cache':{}}
    with torch.inference_mode():
        for k,n,name in ((1024,2048,'q'),(1024,1024,'k_or_v'),(1024,3072,'gate_or_up'),(3072,1024,'down')):
            x=torch.randn(k,device='cuda',dtype=torch.bfloat16); w=torch.randn(k,n,device='cuda',dtype=torch.bfloat16)
            ref=x@w
            for split in (1,2,4):
                got=decode_gemv(x,w,split_k=split); delta=(got.float()-ref.float()).abs()
                out['gemv'].append({'name':name,'shape':[k,n],'split_k':split,'correct':torch.allclose(got,ref,rtol=.03,atol=.03),'max_abs':delta.max().item(),'torch_ms':bench(lambda:x@w),'triton_ms':bench(lambda:decode_gemv(x,w,split_k=split))})
        b,d,capacity=1,128,4096
        q=torch.randn(b,16,d,device='cuda',dtype=torch.bfloat16); k=torch.randn(b,8,d,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
        qw=torch.randn(d,device='cuda',dtype=q.dtype); kw=torch.randn_like(qw); cos=torch.randn(capacity,d//2,device='cuda',dtype=q.dtype); sin=torch.randn_like(cos)
        pos=torch.tensor([2048],device='cuda',dtype=torch.int32); kc=torch.empty(b,8,capacity,d,device='cuda',dtype=q.dtype); vc=torch.empty_like(kc)
        candidate=lambda:fused_qk_norm_rope_cache(q,k,v,qw,kw,cos,sin,pos,kc,vc)
        candidate(); out['qk_cache']={'fused_ms':bench(candidate),'position':2048,'capacity':capacity,'note':'compare against exact framework sequence during model integration'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))


if __name__=='__main__': main()
