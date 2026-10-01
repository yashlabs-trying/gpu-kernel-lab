#!/usr/bin/env python3
"""Compare local Llama RMSNorm candidates with vLLM's active CUDA operations."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from llama_rmsnorm import fused_add_rmsnorm_into, rmsnorm_into  # noqa: E402
from methodology import environment_metadata, percentile  # noqa: E402


def timed_samples(operation, prepare, warmups: int, repetitions: int) -> list[float]:
    for _ in range(warmups):
        prepare()
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def cosine_min(left: torch.Tensor, right: torch.Tensor) -> float:
    rows_left = left.float().reshape(-1, left.shape[-1])
    rows_right = right.float().reshape(-1, right.shape[-1])
    return float(torch.nn.functional.cosine_similarity(rows_left, rows_right, dim=-1).min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 32, 128, 512])
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--warps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.repetitions < 100:
        parser.error("at least 100 repetitions are required for p99")

    from vllm import _custom_ops as vllm_ops

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.manual_seed(1234)
    results = []
    for rows in args.rows:
        x = torch.randn((rows, args.hidden_size), device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
        vllm_out = torch.empty_like(x)
        vllm_ops.rms_norm(vllm_out, x, weight, args.epsilon)

        for warps in args.warps:
            candidate_out = torch.empty_like(x)
            rmsnorm_into(candidate_out, x, weight, args.epsilon, num_warps=warps)
            quality = {
                "max_abs_diff": float((candidate_out.float() - vllm_out.float()).abs().max()),
                "cosine_min": cosine_min(candidate_out, vllm_out),
            }
            candidate_ms = timed_samples(
                lambda: rmsnorm_into(
                    candidate_out, x, weight, args.epsilon, num_warps=warps
                ),
                lambda: None,
                args.warmups,
                args.repetitions,
            )
            baseline_ms = timed_samples(
                lambda: vllm_ops.rms_norm(vllm_out, x, weight, args.epsilon),
                lambda: None,
                args.warmups,
                args.repetitions,
            )
            candidate_latency = latency_summary(candidate_ms)
            baseline_latency = latency_summary(baseline_ms)
            results.append(
                {
                    "operation": "rmsnorm",
                    "rows": rows,
                    "num_warps": warps,
                    "quality": quality,
                    "candidate": candidate_latency,
                    "vllm": baseline_latency,
                    "speedup_vs_vllm": (
                        baseline_latency["median_ms"] / candidate_latency["median_ms"]
                    ),
                    "eligible_for_end_to_end_validation": quality["cosine_min"] >= 0.999
                    and candidate_latency["median_ms"] < baseline_latency["median_ms"],
                    "raw_candidate_ms": candidate_ms,
                    "raw_vllm_ms": baseline_ms,
                }
            )

        candidate_norm = torch.empty_like(x)
        candidate_residual = torch.empty_like(x)
        vllm_x = x.clone()
        vllm_residual = residual.clone()
        vllm_ops.fused_add_rms_norm(vllm_x, vllm_residual, weight, args.epsilon)
        for warps in args.warps:
            fused_add_rmsnorm_into(
                candidate_norm,
                candidate_residual,
                x,
                residual,
                weight,
                args.epsilon,
                num_warps=warps,
            )
            quality = {
                "norm_max_abs_diff": float(
                    (candidate_norm.float() - vllm_x.float()).abs().max()
                ),
                "norm_cosine_min": cosine_min(candidate_norm, vllm_x),
                "residual_max_abs_diff": float(
                    (candidate_residual.float() - vllm_residual.float()).abs().max()
                ),
            }
            candidate_ms = timed_samples(
                lambda: fused_add_rmsnorm_into(
                    candidate_norm,
                    candidate_residual,
                    x,
                    residual,
                    weight,
                    args.epsilon,
                    num_warps=warps,
                ),
                lambda: None,
                args.warmups,
                args.repetitions,
            )
            baseline_ms = timed_samples(
                lambda: vllm_ops.fused_add_rms_norm(
                    vllm_x, vllm_residual, weight, args.epsilon
                ),
                lambda: (vllm_x.copy_(x), vllm_residual.copy_(residual)),
                args.warmups,
                args.repetitions,
            )
            candidate_latency = latency_summary(candidate_ms)
            baseline_latency = latency_summary(baseline_ms)
            results.append(
                {
                    "operation": "fused_add_rmsnorm",
                    "rows": rows,
                    "num_warps": warps,
                    "quality": quality,
                    "candidate": candidate_latency,
                    "vllm": baseline_latency,
                    "speedup_vs_vllm": (
                        baseline_latency["median_ms"] / candidate_latency["median_ms"]
                    ),
                    "eligible_for_end_to_end_validation": quality["norm_cosine_min"] >= 0.999
                    and quality["residual_max_abs_diff"] == 0.0
                    and candidate_latency["median_ms"] < baseline_latency["median_ms"],
                    "raw_candidate_ms": candidate_ms,
                    "raw_vllm_ms": baseline_ms,
                }
            )

    output = {
        "schema_version": 1,
        "benchmark": "llama_rmsnorm_candidate_vs_active_vllm",
        "shape": {"hidden_size": args.hidden_size, "rows": args.rows},
        "dtype": args.dtype,
        "epsilon": args.epsilon,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "environment": environment_metadata(),
        "results": results,
        "microbenchmark_winners": sum(
            item["eligible_for_end_to_end_validation"] for item in results
        ),
        "production_accepted_candidates": 0,
        "production_acceptance_note": (
            "A microbenchmark winner must still improve a matched end-to-end vLLM workload."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
