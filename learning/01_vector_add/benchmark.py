"""Sweep vector and block sizes using correct GPU-timeline timing."""

import argparse
import csv
import statistics
from pathlib import Path

import torch
import triton

from vector_add import vector_add_kernel


def time_cuda(operation, warmup: int, repeats: int, inner_loops: int) -> float:
    """Return median milliseconds per launch using CUDA events."""
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


def bandwidth(size: int, milliseconds: float) -> float:
    # FP32: two 4-byte reads plus one 4-byte write = 12 useful bytes/element.
    return (12 * size) / (milliseconds / 1_000) / 1e9


def run(
    sizes: list[int],
    blocks: list[int],
    warps: list[int],
    warmup: int,
    repeats: int,
):
    rows = []
    for size in sizes:
        torch.manual_seed(0)
        x = torch.randn(size, device="cuda")
        y = torch.randn(size, device="cuda")
        output = torch.empty_like(x)
        inner = 100 if size <= 100_000 else 20

        ms = time_cuda(lambda: torch.add(x, y, out=output), warmup, repeats, inner)
        rows.append(
            ["pytorch", size, "framework", "framework", "framework", ms, bandwidth(size, ms)]
        )

        for block in blocks:
            grid = triton.cdiv(size, block)
            for num_warps in warps:
                launch = lambda b=block, g=grid, w=num_warps: vector_add_kernel[(g,)](
                    x, y, output, size, BLOCK_SIZE=b, num_warps=w
                )
                ms = time_cuda(launch, warmup, repeats, inner)
                rows.append(
                    ["triton", size, block, num_warps, grid, ms, bandwidth(size, ms)]
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 1_000, 100_003, 1_000_000, 10_000_000])
    parser.add_argument("--blocks", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--warps", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    rows = run(args.sizes, args.blocks, args.warps, args.warmup, args.repeats)
    largest_size = max(args.sizes)
    pytorch_ceiling = max(
        row[6] for row in rows if row[0] == "pytorch" and row[1] == largest_size
    )
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: FP32 | useful bytes/element: 12")
    print("The final column uses large-vector PyTorch bandwidth as a measured practical reference.")
    print(
        f"{'kind':<9} {'N':>10} {'block':>10} {'warps':>8} {'grid':>10} "
        f"{'median ms':>12} {'GB/s':>10} {'% ref':>8}"
    )
    for kind, size, block, num_warps, grid, ms, gbps in rows:
        reference_percent = 100 * gbps / pytorch_ceiling
        print(
            f"{kind:<9} {size:>10,d} {str(block):>10} {str(num_warps):>8} "
            f"{str(grid):>10} {ms:>12.6f} {gbps:>10.2f} {reference_percent:>7.1f}%"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "implementation",
                    "size",
                    "block_size",
                    "num_warps",
                    "grid_size",
                    "median_ms",
                    "effective_gbps",
                    "percent_of_measured_pytorch_reference",
                ]
            )
            for row in rows:
                writer.writerow([*row, 100 * row[6] / pytorch_ceiling])
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
