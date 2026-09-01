"""Correctness and hot-microbenchmark gate for the first prefill candidates."""
import argparse
import json
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'learning/03_rmsnorm'), str(ROOT / 'learning/06_gemm')]
from qwen_prefill_gemm import qwen_prefill_gemm
from qwen_rmsnorm import (
    qwen_residual_rmsnorm_reference, qwen_rmsnorm_reference,
    triton_qwen_residual_rmsnorm, triton_qwen_rmsnorm,
)


def timing(fn):
    return float(triton.testing.do_bench(fn, warmup=100, rep=300))


def error(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    return {'max_abs': delta.max().item(), 'mean_abs': delta.mean().item(),
            'allclose': torch.allclose(actual, expected, rtol=.03, atol=.03)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/prefill_candidates.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required')
    torch.manual_seed(123)
    report = {'gpu': torch.cuda.get_device_name(), 'dtype': 'bfloat16',
              'method': 'hot do_bench; 100 ms warmup and 300 ms measurement per callable',
              'rmsnorm': [], 'residual_rmsnorm': [], 'gemm': []}
    with torch.inference_mode():
        for rows, width in ((1,128), (2048,128), (1,1024), (2048,1024)):
            x = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(width, device='cuda', dtype=torch.bfloat16)
            expected = qwen_rmsnorm_reference(x, w)
            actual = triton_qwen_rmsnorm(x, w)
            row = {'shape':[rows,width], **error(actual, expected),
                   'torch_ms':timing(lambda:qwen_rmsnorm_reference(x,w)),
                   'triton_ms':timing(lambda:triton_qwen_rmsnorm(x,w))}
            report['rmsnorm'].append(row)
            residual = torch.randn_like(x)
            expected_y, expected_res = qwen_residual_rmsnorm_reference(x,residual,w)
            actual_y, actual_res = triton_qwen_residual_rmsnorm(x,residual,w)
            row = {'shape':[rows,width], 'normalized':error(actual_y,expected_y),
                   'residual_exact':torch.equal(actual_res,expected_res),
                   'torch_ms':timing(lambda:qwen_residual_rmsnorm_reference(x,residual,w)),
                   'triton_ms':timing(lambda:triton_qwen_residual_rmsnorm(x,residual,w))}
            report['residual_rmsnorm'].append(row)
        for m in (128,512,2048,4096):
            for n,k,label in ((2048,1024,'q'),(1024,1024,'k_or_v'),
                              (3072,1024,'gate_or_up'),(1024,3072,'down')):
                a=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                b=torch.randn(k,n,device='cuda',dtype=torch.bfloat16)
                actual=qwen_prefill_gemm(a,b)
                expected=(a.float()@b.float()).to(a.dtype)
                report['gemm'].append({'name':label,'shape':[m,n,k],**error(actual,expected),
                    'torch_ms':timing(lambda:a@b),'triton_ms':timing(lambda:qwen_prefill_gemm(a,b))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
