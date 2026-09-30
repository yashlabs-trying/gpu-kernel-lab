#!/usr/bin/env python3
"""Production-methodology PyTorch baseline for Llama-3.2-3B.

Request latency is measured with a monotonic clock after CUDA synchronization.
GPU-only prefill/decode latency is measured separately with CUDA events. The
result includes every raw request and token timestamp.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path

import torch
from methodology import RequestSample, aggregate_requests, environment_metadata, write_json
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = "The capital of France is"


def load(model_dir: str, gguf_file: str | None):
    kwargs = {"gguf_file": gguf_file} if gguf_file else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, fix_mistral_regex=False, **kwargs
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        **kwargs,
    )
    model.eval()
    return tokenizer, model


def exact_prompt_ids(tokenizer, prompt: str, input_tokens: int, device) -> torch.Tensor:
    seed = tokenizer(prompt, add_special_tokens=False).input_ids
    if not seed:
        raise ValueError("prompt tokenized to zero tokens")
    ids = (seed * ((input_tokens + len(seed) - 1) // len(seed)))[:input_tokens]
    return torch.tensor([ids], dtype=torch.long, device=device)


def cuda_step(operation):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _prefill(model, input_ids):
    output = model(input_ids, use_cache=True)
    next_id = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
    return output, next_id


def _decode(model, next_id, cache):
    output = model(next_id, past_key_values=cache, use_cache=True)
    following_id = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
    return output, following_id


def run_request(
    request_id: str,
    start_type: str,
    input_ids: torch.Tensor,
    model,
    output_tokens: int,
    synchronize_before: bool = True,
    independent_stream: bool = False,
) -> RequestSample:
    if synchronize_before:
        torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=input_ids.device) if independent_stream else None
    arrival_ns = time.perf_counter_ns()
    token_timestamps: list[int] = []
    decode_gpu_ms: list[float] = []

    stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with stream_context:
        with torch.inference_mode():
            (output, next_id), prefill_gpu_ms = cuda_step(lambda: _prefill(model, input_ids))
            token_timestamps.append(time.perf_counter_ns())
            cache = output.past_key_values

            for _ in range(1, output_tokens):
                (output, next_id), step_gpu_ms = cuda_step(
                    lambda: _decode(model, next_id, cache)
                )
                cache = output.past_key_values
                decode_gpu_ms.append(step_gpu_ms)
                token_timestamps.append(time.perf_counter_ns())

    completed_ns = time.perf_counter_ns()
    return RequestSample(
        request_id=request_id,
        start_type=start_type,
        input_tokens=input_ids.shape[1],
        output_tokens=len(token_timestamps),
        arrival_ns=arrival_ns,
        first_token_ns=token_timestamps[0] if token_timestamps else None,
        token_timestamps_ns=token_timestamps,
        completed_ns=completed_ns,
        gpu_ttft_ms=prefill_gpu_ms,
        gpu_decode_ms=decode_gpu_ms,
    )


def run_request_safe(**kwargs) -> RequestSample:
    fallback_start = time.perf_counter_ns()
    try:
        return run_request(**kwargs)
    except Exception as error:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        completed = time.perf_counter_ns()
        return RequestSample(
            request_id=kwargs["request_id"],
            start_type=kwargs["start_type"],
            input_tokens=int(kwargs["input_ids"].shape[1]),
            output_tokens=0,
            arrival_ns=fallback_start,
            first_token_ns=None,
            token_timestamps_ns=[],
            completed_ns=completed,
            error=f"{type(error).__name__}: {error}",
        )


def run_group(
    *,
    count: int,
    concurrency: int,
    start_type: str,
    input_ids: torch.Tensor,
    model,
    output_tokens: int,
) -> list[RequestSample]:
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_request_safe,
                request_id=f"measured-{index}",
                start_type=start_type,
                input_ids=input_ids,
                model=model,
                output_tokens=output_tokens,
                synchronize_before=concurrency == 1,
                independent_stream=concurrency > 1,
            )
            for index in range(count)
        ]
        samples = [future.result() for future in as_completed(futures)]
    return sorted(samples, key=lambda sample: sample.request_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--gguf-file",
        default=None,
        help="Optional GGUF filename. Omit for a native Hugging Face checkpoint.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.input_tokens < 1 or args.output_tokens < 1 or args.concurrency < 1:
        parser.error("--input-tokens, --output-tokens, and --concurrency must be positive")
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("--repetitions must be positive and --warmups non-negative")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_load_start_ns = time.perf_counter_ns()
    tokenizer, model = load(args.model_dir, args.gguf_file)
    model_load_end_ns = time.perf_counter_ns()
    device = next(model.parameters()).device
    input_ids = exact_prompt_ids(tokenizer, args.prompt, args.input_tokens, device)

    # Cold start is intentionally isolated from both warmups and reported warm metrics.
    cold = run_request("cold-0", "cold", input_ids, model, args.output_tokens)
    if args.warmups:
        run_group(
            count=max(args.warmups, args.concurrency),
            concurrency=args.concurrency,
            start_type="warmup",
            input_ids=input_ids,
            model=model,
            output_tokens=args.output_tokens,
        )

    environment_before = environment_metadata()
    torch.cuda.reset_peak_memory_stats()
    benchmark_start_ns = time.perf_counter_ns()
    samples = run_group(
        count=args.repetitions,
        concurrency=args.concurrency,
        start_type="warm",
        input_ids=input_ids,
        model=model,
        output_tokens=args.output_tokens,
    )
    benchmark_end_ns = time.perf_counter_ns()

    result = {
        "schema_version": 2,
        "benchmark": "llama_pytorch_concurrent",
        "engine": "transformers",
        "model": args.model_dir,
        "dtype": "float16",
        "batch_size": 1,
        "concurrency": args.concurrency,
        "prompt": {
            "source_text": args.prompt,
            "input_tokens": args.input_tokens,
            "token_ids": input_ids[0].tolist(),
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
            "request_clock": "time.perf_counter_ns (monotonic)",
            "gpu_clock": "torch.cuda.Event with end-event synchronization",
            "concurrency_execution": "one CUDA stream per request when concurrency > 1",
            "p99_minimum_repetitions": 100,
            "cold_start_excluded_from_warm_aggregates": True,
            "cold_start_scope": "model load plus first inference are reported separately",
        },
        "model_load_seconds": (model_load_end_ns - model_load_start_ns) / 1_000_000_000,
        "environment_before": environment_before,
        "cold_start": cold.to_dict(),
        "aggregate": aggregate_requests(
            samples,
            benchmark_start_ns=benchmark_start_ns,
            benchmark_end_ns=benchmark_end_ns,
        ),
        "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(0),
        "environment_after": environment_metadata(),
        "raw_samples": [sample.to_dict() for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output), result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
