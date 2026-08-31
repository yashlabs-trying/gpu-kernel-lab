"""Diagnose a BF16 mismatch using FP32 reference and library reduction policy."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/06_gemm'))
from gemm import triton_gemm


def main():
    torch.manual_seed(0)
    dtype=torch.bfloat16
    # Reproduce the RNG sequence used by compare_native exactly.
    for tokens in (1,512):
        x=torch.randn((tokens,1024),device='cuda',dtype=dtype)
        w=torch.randn(1024,device='cuda',dtype=dtype)
        gate=torch.randn((tokens,3072),device='cuda',dtype=dtype)
        up=torch.randn_like(gate)
        a=torch.randn((tokens,1024),device='cuda',dtype=dtype)
        b=torch.randn((1024,2048),device='cuda',dtype=dtype)
    original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False
    reference=(a.float()@b.float()).to(dtype)
    outputs={'triton':triton_gemm(a,b),'pytorch_default':a@b}
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    outputs['pytorch_no_reduced_precision']=a@b
    rows={}
    for name,value in outputs.items():
        error=(value.float()-reference.float()).abs()
        rows[name]={'max_abs_error_vs_fp32_rounded':error.max().item(),
                    'mean_abs_error_vs_fp32_rounded':error.mean().item(),
                    'different_elements':(value!=reference).sum().item(),
                    'allclose_003':torch.allclose(value,reference,rtol=.03,atol=.03)}
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
    result={'shape':[512,2048,1024],'original_allow_bf16_reduced_precision_reduction':original,
            'reference':'FP32 matmul with TF32 disabled, rounded to BF16','results':rows}
    (ROOT/'results/profiling_20260831/gemm_accuracy_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
