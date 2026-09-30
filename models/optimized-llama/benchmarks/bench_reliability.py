#!/usr/bin/env python3
"""Soak and overload harness for a running vLLM server."""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from bench_serving import exact_prompt_ids, run_group
from methodology import aggregate_requests, environment_metadata, write_json
from transformers import AutoTokenizer


def fetch_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llama-3.2-3b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--mode", choices=("soak", "overload"), required=True)
    parser.add_argument("--duration-seconds", type=float, default=3600.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=0)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.concurrency <= 0:
        parser.error("duration and concurrency must be positive")

    base_url = args.base_url.rstrip("/")
    health_before = fetch_text(base_url + "/health")
    metrics_before = fetch_text(base_url + "/metrics")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt_ids = exact_prompt_ids(tokenizer, "The capital of France is", args.input_tokens)
    common = {
        "start_type": "warm",
        "endpoint": base_url + "/v1/completions",
        "api_key": "EMPTY",
        "model": args.model,
        "prompt_token_ids": prompt_ids,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "timeout": args.timeout,
    }

    warmup_samples = run_group(args.concurrency, args.concurrency, **common)
    if any(sample.error for sample in warmup_samples):
        raise SystemExit("reliability warmup failed")
    samples = []
    environment_before = environment_metadata()
    benchmark_start_ns = time.perf_counter_ns()
    deadline = time.monotonic() + args.duration_seconds
    cycle = 0
    if args.mode == "overload":
        target = args.requests or max(100, args.concurrency * 4)
        samples = run_group(target, args.concurrency, **common)
    else:
        while time.monotonic() < deadline:
            group = run_group(args.concurrency, args.concurrency, **common)
            for sample in group:
                sample.request_id = f"cycle-{cycle}-{sample.request_id}"
            samples.extend(group)
            cycle += 1
    benchmark_end_ns = time.perf_counter_ns()
    metrics_after = fetch_text(base_url + "/metrics")
    health_after = fetch_text(base_url + "/health")

    result = {
        "schema_version": 2,
        "benchmark": f"llama_vllm_{args.mode}",
        "model": args.model,
        "engine": "vllm-openai-server",
        "concurrency": args.concurrency,
        "methodology": {
            "warmups": len(warmup_samples),
            "repetitions": len(samples),
            "duration_target_seconds": args.duration_seconds if args.mode == "soak" else None,
            "mode": args.mode,
        },
        "environment_before": environment_before,
        "environment_after": environment_metadata(),
        "server_health_before": health_before,
        "server_health_after": health_after,
        "server_metrics_before": metrics_before,
        "server_metrics_after": metrics_after,
        "aggregate": aggregate_requests(
            samples,
            benchmark_start_ns=benchmark_start_ns,
            benchmark_end_ns=benchmark_end_ns,
        ),
        "raw_samples": [sample.to_dict() for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output), result)
    print(args.output)
    if args.mode == "soak" and result["aggregate"]["failed_requests"]:
        raise SystemExit(f"{result['aggregate']['failed_requests']} requests failed")


if __name__ == "__main__":
    main()
