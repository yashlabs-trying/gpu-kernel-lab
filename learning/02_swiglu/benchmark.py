"""Benchmark eager PyTorch versus fused Triton SwiGLU."""

import argparse
import csv
import statistics
from pathlib import Path

import torch
import torch.nn.functional as torch_functional
import triton

from swiglu import swiglu_kernel


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def time_cuda(operation, warmup: int, repeats: int, inner_loops: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_loops):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner_loops)
    return statistics.median(samples)


def useful_bandwidth(size: int, element_size: int, milliseconds: float) -> float:
    # Ideal fused traffic: gate read + up read + output write = 3 elements.
    return (3 * size * element_size) / (milliseconds / 1_000) / 1e9


def run(args):
    dtype = DTYPES[args.dtype]
    element_size = torch.empty((), dtype=dtype).element_size()
    rows = []
    for size in args.sizes:
        torch.manual_seed(0)
        gate = torch.randn(size, device="cuda", dtype=dtype)
        up = torch.randn(size, device="cuda", dtype=dtype)
        output = torch.empty_like(gate)
        inner = 100 if size <= 100_000 else 20

        # This eager expression normally creates a SiLU intermediate and uses two launches.
        eager = lambda: torch_functional.silu(gate) * up
        ms = time_cuda(eager, args.warmup, args.repeats, inner)
        rows.append(["pytorch_eager", size, "framework", "framework", "framework", ms,
                     useful_bandwidth(size, element_size, ms)])

        for block in args.blocks:
            grid = triton.cdiv(size, block)
            for warps in args.warps:
                launch = lambda b=block, g=grid, w=warps: swiglu_kernel[(g,)](
                    gate, up, output, size, BLOCK_SIZE=b, num_warps=w
                )
                ms = time_cuda(launch, args.warmup, args.repeats, inner)
                rows.append(["triton_fused", size, block, warps, grid, ms,
                             useful_bandwidth(size, element_size, ms)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[128, 1_000, 100_003, 1_000_000, 10_000_000])
    parser.add_argument("--blocks", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--warps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    rows = run(args)
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {args.dtype}")
    print(f"{'kind':<15} {'N':>10} {'block':>10} {'warps':>8} {'grid':>10} {'ms':>12} {'useful GB/s':>13}")
    for kind, size, block, warps, grid, ms, gbps in rows:
        print(f"{kind:<15} {size:>10,d} {str(block):>10} {str(warps):>8} "
              f"{str(grid):>10} {ms:>12.6f} {gbps:>13.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["implementation", "size", "block_size", "num_warps",
                             "grid_size", "median_ms", "useful_gbps"])
            writer.writerows(rows)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
