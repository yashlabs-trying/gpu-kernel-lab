"""Sweep row width, block size, and warps for standalone softmax."""
import argparse, csv, statistics
from pathlib import Path
import torch, triton
from softmax import softmax_kernel


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
    p.add_argument("--shapes", nargs="+", default=["128x128", "128x1000", "128x3000", "128x4096"])
    p.add_argument("--warps", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--output", type=Path); args = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required")
    dtype = getattr(torch, args.dtype); result = []
    for shape_text in args.shapes:
        rows, columns = map(int, shape_text.split("x"))
        x, output = torch.randn((rows, columns), device="cuda", dtype=dtype), torch.empty((rows, columns), device="cuda", dtype=dtype)
        result.append(["pytorch", rows, columns, "framework", "framework", timing(lambda: torch.softmax(x.float(), -1).to(dtype))])
        minimum = triton.next_power_of_2(columns)
        blocks = [minimum] + ([minimum * 2] if minimum * 2 <= 65_536 else [])
        for block in blocks:
            for warps in args.warps:
                launch = lambda b=block, w=warps: softmax_kernel[(rows,)](x, output, columns, BLOCK_SIZE=b, num_warps=w)
                result.append(["triton", rows, columns, block, warps, timing(launch)])
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype: {args.dtype}")
    for row in result: print(row)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["implementation", "rows", "columns", "block", "warps", "median_ms"]); w.writerows(result)


if __name__ == "__main__": main()

