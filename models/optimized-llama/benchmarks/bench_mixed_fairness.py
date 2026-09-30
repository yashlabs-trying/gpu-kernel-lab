#!/usr/bin/env python3
"""Measure short/long request fairness under one shared vLLM scheduler load."""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bench_serving import exact_prompt_ids, stream_request
from methodology import aggregate_requests, environment_metadata, write_json
from transformers import AutoTokenizer


def workload_labels(repetitions_per_class: int, seed: int) -> list[str]:
    labels = ["short"] * repetitions_per_class + ["long"] * repetitions_per_class
    random.Random(seed).shuffle(labels)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--short-input-tokens", type=int, default=128)
    parser.add_argument("--long-input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repetitions-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "short_input_tokens",
        "long_input_tokens",
        "output_tokens",
        "concurrency",
        "repetitions_per_class",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.short_input_tokens >= args.long_input_tokens:
        parser.error("short input length must be smaller than long input length")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, fix_mistral_regex=False)
    prompts = {
        "short": exact_prompt_ids(tokenizer, "The capital of France is", args.short_input_tokens),
        "long": exact_prompt_ids(tokenizer, "The capital of France is", args.long_input_tokens),
    }
    common = {
        "start_type": "warm",
        "endpoint": args.base_url.rstrip("/") + "/v1/completions",
        "api_key": args.api_key,
        "model": args.model,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "timeout": args.timeout,
    }
    for label in ("short", "long"):
        sample = stream_request(
            request_id=f"warmup-{label}", prompt_token_ids=prompts[label], **common
        )
        if sample.error:
            raise SystemExit(f"{label} warmup failed: {sample.error}")

    environment_before = environment_metadata()
    labels = workload_labels(args.repetitions_per_class, args.seed)
    start_ns = time.perf_counter_ns()
    samples = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                stream_request,
                request_id=f"{label}-{index:04d}",
                prompt_token_ids=prompts[label],
                **common,
            ): label
            for index, label in enumerate(labels)
        }
        for future in as_completed(futures):
            samples.append(future.result())
    end_ns = time.perf_counter_ns()
    cohorts = {
        label: aggregate_requests(
            [sample for sample in samples if sample.request_id.startswith(label + "-")],
            benchmark_start_ns=start_ns,
            benchmark_end_ns=end_ns,
        )
        for label in ("short", "long")
    }
    short_p99 = cohorts["short"]["request_latency"]["end_to_end"]["p99_ms"]
    long_p99 = cohorts["long"]["request_latency"]["end_to_end"]["p99_ms"]
    result = {
        "schema_version": 2,
        "benchmark": "llama_vllm_mixed_length_fairness",
        "engine": "vllm-openai-server",
        "model": args.model,
        "concurrency": args.concurrency,
        "workload": {
            "short_input_tokens": args.short_input_tokens,
            "long_input_tokens": args.long_input_tokens,
            "output_tokens": args.output_tokens,
            "repetitions_per_class": args.repetitions_per_class,
            "submission_order": labels,
            "seed": args.seed,
        },
        "methodology": {
            "mixed_in_one_executor": True,
            "warmups_per_class": 1,
            "p99_minimum_repetitions_per_class": 100,
        },
        "environment_before": environment_before,
        "aggregate": aggregate_requests(
            samples, benchmark_start_ns=start_ns, benchmark_end_ns=end_ns
        ),
        "cohorts": cohorts,
        "fairness": {
            "long_to_short_p99_end_to_end_ratio": (
                long_p99 / short_p99 if short_p99 and long_p99 else None
            )
        },
        "environment_after": environment_metadata(),
        "raw_samples": [sample.to_dict() for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output), result)
    print(args.output)
    if result["aggregate"]["failed_requests"]:
        raise SystemExit(f"{result['aggregate']['failed_requests']} requests failed")


if __name__ == "__main__":
    main()
