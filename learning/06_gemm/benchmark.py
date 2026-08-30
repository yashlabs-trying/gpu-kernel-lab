"""Benchmark GEMM tile, warp, and pipeline configurations."""
import argparse, csv, statistics
from pathlib import Path
import torch, triton
from gemm import gemm_kernel

CONFIGS = [
    (32, 32, 32, 4, 2), (64, 64, 32, 4, 2), (64, 64, 32, 4, 3),
    (64, 64, 32, 4, 4), (128, 64, 32, 8, 3), (64, 128, 32, 8, 3),
    (128, 128, 32, 8, 4), (64, 128, 64, 8, 4),
]


def timing(fn, warmup=10, repeats=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); values = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize(); values.append(a.elapsed_time(b))
    return statistics.median(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["256x256x256", "1024x1024x1024", "4096x4096x1024"])
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--output", type=Path); args = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    dtype = getattr(torch, args.dtype); rows = []
    for text in args.shapes:
        m, n, k = map(int, text.split("x")); torch.manual_seed(0)
        a, b, c = torch.randn((m, k), device="cuda", dtype=dtype), torch.randn((k, n), device="cuda", dtype=dtype), torch.empty((m, n), device="cuda", dtype=dtype)
        ms = timing(lambda: torch.mm(a, b)); rows.append(["pytorch", text, "framework", "framework", "framework", "framework", "framework", ms, 2*m*n*k/(ms/1000)/1e12])
        for bm, bn, bk, warps, stages in CONFIGS:
            grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
            launch = lambda: gemm_kernel[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=8, num_warps=warps, num_stages=stages)
            ms = timing(launch); rows.append(["triton", text, bm, bn, bk, warps, stages, ms, 2*m*n*k/(ms/1000)/1e12])
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {args.dtype}")
    for row in rows: print(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["implementation","MxNxK","BM","BN","BK","warps","stages","median_ms","TFLOP/s"]); w.writerows(rows)


if __name__ == "__main__": main()
