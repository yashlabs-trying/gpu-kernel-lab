"""Capture representative teaching-kernel launches and GEMM codegen evidence."""
import json
import sys
from pathlib import Path
import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / 'learning').glob('0*'):
    sys.path.insert(0, str(folder))
from vector_add import triton_add
from swiglu import triton_swiglu
from rmsnorm import triton_rmsnorm, triton_residual_rmsnorm
from rope import triton_rope_qk, build_rope_tables
from softmax import triton_softmax
from gemm import gemm_kernel
from attention import triton_attention


def main():
    out = ROOT / 'results/profiling_20260831'
    device, dtype = 'cuda', torch.bfloat16
    with torch.inference_mode():
        vector = torch.randn(1_000_000, device=device)
        x = torch.randn((512, 1024), device=device, dtype=dtype)
        w = torch.ones(1024, device=device, dtype=dtype)
        gate = torch.randn((512, 3072), device=device, dtype=dtype)
        q = torch.randn((1, 4, 512, 128), device=device, dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        cos, sin = build_rope_tables(512,128,q.device,dtype)
        lengths = torch.tensor([512], device=device, dtype=torch.int32)
        b = torch.randn((1024,2048), device=device, dtype=dtype)
        c = torch.empty((512,2048), device=device, dtype=dtype)
        def gemm():
            return gemm_kernel[(triton.cdiv(512,64)*triton.cdiv(2048,64),)](
                x,b,c,512,2048,1024,1024,1,2048,1,2048,1,
                BLOCK_M=64,BLOCK_N=64,BLOCK_K=32,GROUP_SIZE_M=8,
                num_warps=4,num_stages=3)
        compiled = gemm()
        ptx = compiled.asm['ptx']
        metadata = {'registers': compiled.n_regs, 'spill_count': compiled.n_spills,
                    'shared_bytes': compiled.metadata.shared,
                    'cp_async_mentions': ptx.count('cp.async'),
                    'mma_sync_mentions': ptx.count('mma.sync'),
                    'meaning': 'Compiler metadata and instruction presence, not runtime achieved occupancy/stall counters.'}
        (out / 'triton_gemm_codegen.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (out / 'triton_gemm.ptx').write_text(ptx)
        operations = {
            'vector_add_1M': lambda: triton_add(vector,vector),
            'swiglu_512x3072': lambda: triton_swiglu(gate,gate),
            'rmsnorm_512x1024': lambda: triton_rmsnorm(x,w),
            'residual_rmsnorm_512x1024': lambda: triton_residual_rmsnorm(x,x,w),
            'rope_1x4x512x128': lambda: triton_rope_qk(q,k,cos,sin),
            'softmax_512x1024': lambda: triton_softmax(x),
            'gemm_512x2048x1024': gemm,
            'attention_1x4x512x128': lambda: triton_attention(q,k,v,lengths,lengths),
        }
        for fn in operations.values():
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for name, fn in operations.items():
            torch.cuda.nvtx.range_push(name)
            fn()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    main()
