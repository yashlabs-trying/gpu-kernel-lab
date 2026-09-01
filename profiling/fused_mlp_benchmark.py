"""Correctness and hot timings for gate+up+SwiGLU fusion."""
import argparse,json,sys
from pathlib import Path
import torch,triton
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/02_swiglu'))
from fused_projection_swiglu import fused_projection_swiglu


def eager(x,wg,wu): return F.silu(F.linear(x,wg))*F.linear(x,wu)
def bench(fn): return float(triton.testing.do_bench(fn,warmup=200,rep=500))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=ROOT/'results/fused_mlp.json'); a=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU required')
    torch.manual_seed(12); rows=[]
    with torch.inference_mode():
        wg=torch.randn(3072,1024,device='cuda',dtype=torch.bfloat16); wu=torch.randn_like(wg)
        for m in (1,16,128,512,2048,4096):
            x=torch.randn((m,1024),device='cuda',dtype=torch.bfloat16)
            candidate=fused_projection_swiglu(x[0] if m==1 else x,wg,wu)
            expected=eager(x[0] if m==1 else x,wg,wu); delta=(candidate.float()-expected.float()).abs()
            rows.append({'m':m,'shape':[m,1024,3072],
                'correct':torch.allclose(candidate,expected,rtol=.03,atol=.03),
                'max_abs':delta.max().item(),'mean_abs':delta.mean().item(),
                'eager_ms':bench(lambda:eager(x[0] if m==1 else x,wg,wu)),
                'fused_ms':bench(lambda:fused_projection_swiglu(x[0] if m==1 else x,wg,wu))})
    report={'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','rows':rows,
            'warning':'hot microbenchmark; inspect registers/spills and run full-model validation'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
