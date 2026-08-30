"""Benchmark separate PyTorch Q/K RoPE against one-launch Triton RoPE."""
import argparse, csv, statistics
from pathlib import Path
import torch, triton
from rope import build_rope_tables, pytorch_rope, rope_qk_kernel


def timing(fn, warmup=10, repeats=20, inner=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); values = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner): fn()
        b.record(); b.synchronize(); values.append(a.elapsed_time(b) / inner)
    return statistics.median(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["1x8x128x128", "1x8x2048x128"])
    p.add_argument("--warps", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--output", type=Path); args = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    dtype = getattr(torch, args.dtype); rows = []
    for shape_text in args.shapes:
        batch, q_heads, sequence, dimension = map(int, shape_text.split("x")); k_heads = max(1, q_heads // 4)
        q = torch.randn((batch, q_heads, sequence, dimension), device="cuda", dtype=dtype)
        k = torch.randn((batch, k_heads, sequence, dimension), device="cuda", dtype=dtype)
        cos, sin = build_rope_tables(sequence, dimension, q.device, dtype)
        ms = timing(lambda: (pytorch_rope(q, cos, sin), pytorch_rope(k, cos, sin)))
        rows.append(["pytorch_separate", shape_text, k_heads, "framework", "framework", ms])
        pair_block = triton.next_power_of_2(dimension // 2)
        for warps in args.warps:
            qo, ko = torch.empty_like(q), torch.empty_like(k)
            q_rows, k_rows = q.numel() // dimension, k.numel() // dimension
            launch = lambda w=warps: rope_qk_kernel[(q_rows + k_rows,)](
                q, k, cos, sin, qo, ko, q_rows, sequence, dimension,
                PAIR_BLOCK=pair_block, num_warps=w)
            ms = timing(launch); rows.append(["triton_qk_one_launch", shape_text, k_heads, pair_block, warps, ms])
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {args.dtype}")
    for row in rows: print(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["implementation", "BHQSD", "k_heads", "pair_block", "warps", "median_ms"]); w.writerows(rows)


if __name__ == "__main__": main()

