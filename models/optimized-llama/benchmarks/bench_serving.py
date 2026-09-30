#!/usr/bin/env python3
"""Benchmark a real vLLM OpenAI-compatible server using streamed token IDs.

The server must be started separately. For matched results, disable prefix
caching and use the same model, token IDs, output length, and sampling settings
as ``bench_torch_baseline.py``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from methodology import RequestSample, aggregate_requests, environment_metadata, write_json
from transformers import AutoTokenizer

DEFAULT_PROMPT = "The capital of France is"


def exact_prompt_ids(tokenizer, prompt: str, input_tokens: int) -> list[int]:
    seed = tokenizer(prompt, add_special_tokens=False).input_ids
    if not seed:
        raise ValueError("prompt tokenized to zero tokens")
    return (seed * ((input_tokens + len(seed) - 1) // len(seed)))[:input_tokens]


def consume_sse(
    lines: Iterable[bytes], *, clock_ns: Callable[[], int] = time.perf_counter_ns
) -> tuple[list[int], list[int], bool]:
    """Extract delta token IDs and client-receipt times from a vLLM SSE stream."""
    output_ids: list[int] = []
    timestamps: list[int] = []
    multi_token_chunk = False
    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        token_ids = choices[0].get("token_ids") or []
        received_ns = clock_ns()
        multi_token_chunk = multi_token_chunk or len(token_ids) > 1
        output_ids.extend(int(token_id) for token_id in token_ids)
        timestamps.extend([received_ns] * len(token_ids))
    return output_ids, timestamps, multi_token_chunk


def stream_request(
    *,
    request_id: str,
    start_type: str,
    endpoint: str,
    api_key: str,
    model: str,
    prompt_token_ids: list[int],
    output_tokens: int,
    seed: int,
    timeout: float,
) -> RequestSample:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "temperature": 0.0,
        "seed": seed,
        "ignore_eos": True,
        "stream": True,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    arrival_ns = time.perf_counter_ns()
    timestamps: list[int] = []
    output_ids: list[int] = []
    error: str | None = None
    multi_token_chunk = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            output_ids, timestamps, multi_token_chunk = consume_sse(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed_ns = time.perf_counter_ns()
    if error is None and len(output_ids) != output_tokens:
        error = (
            f"expected {output_tokens} streamed token IDs, received {len(output_ids)}; "
            "verify that this vLLM version supports return_token_ids"
        )
    if error is None and multi_token_chunk:
        error = (
            "server returned multiple token IDs in one stream chunk; exact per-token "
            "ITL is unavailable (launch with stream interval 1)"
        )
    return RequestSample(
        request_id=request_id,
        start_type=start_type,
        input_tokens=len(prompt_token_ids),
        output_tokens=len(output_ids),
        arrival_ns=arrival_ns,
        first_token_ns=timestamps[0] if timestamps else None,
        token_timestamps_ns=timestamps,
        completed_ns=completed_ns,
        output_token_ids=output_ids,
        error=error,
    )


def run_group(count: int, concurrency: int, **kwargs) -> list[RequestSample]:
    samples: list[RequestSample] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(stream_request, request_id=f"measured-{index}", **kwargs)
            for index in range(count)
        ]
        for future in as_completed(futures):
            samples.append(future.result())
    return sorted(samples, key=lambda sample: sample.request_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True, help="Model name exposed by /v1/models")
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer path or HF ID")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--cuda-graphs", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in ("input_tokens", "output_tokens", "concurrency", "repetitions"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, fix_mistral_regex=False
    )
    prompt_token_ids = exact_prompt_ids(tokenizer, args.prompt, args.input_tokens)
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    common = {
        "start_type": "warm",
        "endpoint": endpoint,
        "api_key": args.api_key,
        "model": args.model,
        "prompt_token_ids": prompt_token_ids,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "timeout": args.timeout,
    }

    cold = stream_request(request_id="cold-0", **{**common, "start_type": "cold"})
    if cold.error:
        raise SystemExit(f"cold request failed: {cold.error}")
    for index in range(args.warmups):
        warmup = stream_request(request_id=f"warmup-{index}", **common)
        if warmup.error:
            raise SystemExit(f"warmup request failed: {warmup.error}")

    environment_before = environment_metadata()
    benchmark_start_ns = time.perf_counter_ns()
    samples = run_group(args.repetitions, args.concurrency, **common)
    benchmark_end_ns = time.perf_counter_ns()
    result = {
        "schema_version": 2,
        "benchmark": "llama_vllm_openai_streaming",
        "engine": "vllm-openai-server",
        "model": args.model,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "cuda_graphs": args.cuda_graphs == "enabled",
        "attention_backend": args.attention_backend,
        "tensor_parallel_size": args.tensor_parallel_size,
        "endpoint": endpoint,
        "batch_size": None,
        "concurrency": args.concurrency,
        "prompt": {
            "source_text": args.prompt,
            "input_tokens": len(prompt_token_ids),
            "token_ids": prompt_token_ids,
        },
        "sampling": {
            "strategy": "greedy",
            "temperature": 0.0,
            "seed": args.seed,
            "ignore_eos": True,
            "output_tokens": args.output_tokens,
        },
        "methodology": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "request_clock": "time.perf_counter_ns (monotonic client receipt)",
            "token_timing": "one timestamp per returned vLLM delta token ID",
            "p99_minimum_repetitions": 100,
            "cold_start_excluded_from_warm_aggregates": True,
            "cold_start_scope": (
                "first request to a pre-launched server; server model load excluded"
            ),
            "server_requirement": "launch with prefix caching disabled for matched runs",
        },
        "environment_before": environment_before,
        "cold_start": cold.to_dict(),
        "aggregate": aggregate_requests(
            samples,
            benchmark_start_ns=benchmark_start_ns,
            benchmark_end_ns=benchmark_end_ns,
        ),
        "environment_after": environment_metadata(),
        "raw_samples": [sample.to_dict() for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output), result)
    print(f"Wrote {args.output}")
    if result["aggregate"]["failed_requests"]:
        raise SystemExit(f"{result['aggregate']['failed_requests']} measured requests failed")


if __name__ == "__main__":
    main()
