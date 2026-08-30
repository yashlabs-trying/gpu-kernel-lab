"""Tune RMSNorm block/warp choices and measure residual fusion."""

import argparse
import csv
import statistics
from pathlib import Path

import torch
import triton

from rmsnorm import (
    pytorch_residual_rmsnorm,
    pytorch_rmsnorm,
    residual_rmsnorm_kernel,
    rmsnorm_kernel,
)


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def time_cuda(operation, warmup: int, repeats: int, inner: int = 20) -> float:
    for _ in range(warmup): operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner): operation()
        end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) / inner)
    return statistics.median(samples)


def run(args):
    dtype = DTYPES[args.dtype]
    rows = []
    for row_count, hidden in args.shapes:
        torch.manual_seed(0)
        x = torch.randn((row_count, hidden), device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden, device="cuda", dtype=dtype)
        output, residual_output = torch.empty_like(x), torch.empty_like(x)
        minimum_block = triton.next_power_of_2(hidden)
        blocks = [minimum_block]
        if minimum_block * 2 <= 65_536:
            blocks.append(minimum_block * 2)

        ms = time_cuda(lambda: pytorch_rmsnorm(x, weight, args.epsilon), args.warmup, args.repeats)
        rows.append(["pytorch_rmsnorm", row_count, hidden, "framework", "framework", ms])
        ms = time_cuda(
            lambda: pytorch_residual_rmsnorm(x, residual, weight, args.epsilon),
            args.warmup,
            args.repeats,
        )
        rows.append(["pytorch_residual_norm", row_count, hidden, "framework", "framework", ms])

        for block in blocks:
            for warps in args.warps:
                standalone = lambda b=block, w=warps: rmsnorm_kernel[(row_count,)](
                    x, weight, output, hidden, epsilon=args.epsilon,
                    BLOCK_SIZE=b, num_warps=w,
                )
                ms = time_cuda(standalone, args.warmup, args.repeats)
                rows.append(["triton_rmsnorm", row_count, hidden, block, warps, ms])

                fused = lambda b=block, w=warps: residual_rmsnorm_kernel[(row_count,)](
                    x, residual, weight, output, residual_output, hidden,
                    epsilon=args.epsilon, BLOCK_SIZE=b, num_warps=w,
                )
                ms = time_cuda(fused, args.warmup, args.repeats)
                rows.append(["triton_fused_residual", row_count, hidden, block, warps, ms])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", type=lambda s: tuple(map(int, s.split("x"))),
                        default=[(128, 256), (128, 1_024), (128, 4_096)])
    parser.add_argument("--warps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    rows = run(args)
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {args.dtype}")
    print(f"{'implementation':<24} {'rows':>7} {'hidden':>8} {'block':>8} {'warps':>7} {'median ms':>12}")
    for kind, row_count, hidden, block, warps, ms in rows:
        print(f"{kind:<24} {row_count:>7} {hidden:>8} {str(block):>8} {str(warps):>7} {ms:>12.6f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["implementation", "rows", "hidden_size", "block_size", "num_warps", "median_ms"])
            writer.writerows(rows)


if __name__ == "__main__": main()
