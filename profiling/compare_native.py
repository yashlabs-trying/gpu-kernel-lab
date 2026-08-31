"""Supplement teaching benchmarks with native baselines and CUDA-graph timing.

Graphs reduce Python launch starvation; these are hot, repeated microbenchmarks,
not end-to-end model timings or cold-DRAM bandwidth measurements.
"""
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / 'learning').glob('0*'):
    sys.path.insert(0, str(folder))
from swiglu import pytorch_swiglu, triton_swiglu
from rmsnorm import pytorch_rmsnorm, triton_rmsnorm
from softmax import pytorch_softmax, triton_softmax
from gemm import triton_gemm
from attention import triton_attention, pytorch_attention


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    times = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 16)
    return {'median_ms': statistics.median(times), 'samples_ms': times}


def main():
    strict = os.environ.get('KERNELLAB_STRICT_GEMM') == '1'
    if strict:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    filename = 'native_comparison_strict.json' if strict else 'native_comparison.json'
    output = ROOT / 'results/profiling_20260831' / filename
    report = {'method': '16 calls per CUDA graph replay, 10 samples; allocation patterns captured; hot data',
              'dtype': 'bfloat16', 'gpu': torch.cuda.get_device_name(),
              'allow_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              'results': []}
    torch.manual_seed(0)
    dtype = torch.bfloat16

    def evaluate(case, reference_fn, implementations):
        reference = reference_fn()
        for name, fn in implementations.items():
            row = {'case': case, 'implementation': name}
            try:
                actual = fn()
                torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
                row['max_abs_error'] = (actual.float() - reference.float()).abs().max().item()
                row['correct'] = True
                row.update(timed(fn))
            except Exception as exc:
                row.update({'correct': False, 'error': str(exc)})
            report['results'].append(row)
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)

    with torch.inference_mode():
        for tokens in (1, 512, 2048):
            x = torch.randn((tokens, 1024), device='cuda', dtype=dtype)
            w = torch.randn(1024, device='cuda', dtype=dtype)
            evaluate(f'rmsnorm_{tokens}x1024', lambda: pytorch_rmsnorm(x,w,1e-6), {
                'pytorch_eager': lambda: pytorch_rmsnorm(x,w,1e-6),
                'pytorch_native': lambda: F.rms_norm(x,(1024,),w,1e-6),
                'triton': lambda: triton_rmsnorm(x,w),
            })
            gate = torch.randn((tokens,3072), device='cuda', dtype=dtype)
            up = torch.randn_like(gate)
            evaluate(f'swiglu_{tokens}x3072', lambda: pytorch_swiglu(gate,up), {
                'pytorch_eager': lambda: pytorch_swiglu(gate,up),
                'triton': lambda: triton_swiglu(gate,up),
            })
            a = torch.randn((tokens,1024), device='cuda', dtype=dtype)
            b = torch.randn((1024,2048), device='cuda', dtype=dtype)
            evaluate(f'q_projection_{tokens}x2048x1024', lambda: a@b, {
                'pytorch_mm': lambda: a@b,
                'triton_default': lambda: triton_gemm(a,b),
            })
        for columns in (128, 1000, 3000):
            x = torch.randn((128, columns),device='cuda',dtype=dtype)
            evaluate(f'softmax_128x{columns}', lambda: pytorch_softmax(x), {
                'pytorch_eager_fp32': lambda: pytorch_softmax(x),
                'pytorch_native': lambda: torch.softmax(x,dim=-1),
                'triton': lambda: triton_softmax(x),
            })
        for sequence in (128,512):
            q = torch.randn((1,4,sequence,128),device='cuda',dtype=dtype)
            k,v = torch.randn_like(q),torch.randn_like(q)
            lengths = torch.tensor([sequence],device='cuda',dtype=torch.int32)
            evaluate(f'attention_1x4x{sequence}x128', lambda: pytorch_attention(q,k,v,lengths,lengths,True)[0], {
                'pytorch_materialized': lambda: pytorch_attention(q,k,v,lengths,lengths,True)[0],
                'pytorch_sdpa': lambda: F.scaled_dot_product_attention(q,k,v,is_causal=True),
                'triton': lambda: triton_attention(q,k,v,lengths,lengths)[0],
            })


if __name__ == '__main__':
    main()
