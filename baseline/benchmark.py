#!/usr/bin/env python3
"""Reproducible Qwen prefill and decode baseline for KernelLab."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def cuda_elapsed_ms(operation) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def make_inputs(tokenizer, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    seed = tokenizer.encode(
        "GPU kernels move data and perform parallel arithmetic efficiently. ",
        add_special_tokens=False,
    )
    if not seed:
        raise RuntimeError("Tokenizer returned no seed tokens.")
    values = (seed * ((sequence_length + len(seed) - 1) // len(seed)))[:sequence_length]
    input_ids = torch.tensor([values], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": round(statistics.median(samples_ms), 4),
        "p20_ms": round(percentile(samples_ms, 0.20), 4),
        "p80_ms": round(percentile(samples_ms, 0.80), 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
    }


def benchmark_prefill(model, tokenizer, sequence_length: int, warmup: int, repeats: int):
    input_ids, attention_mask = make_inputs(tokenizer, sequence_length)

    def operation():
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples = [cuda_elapsed_ms(operation) for _ in range(repeats)]
    summary = summarize(samples)
    summary.update(
        {
            "sequence_length": sequence_length,
            "tokens_per_second": round(sequence_length / (summary["median_ms"] / 1000), 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        }
    )
    return summary


def benchmark_decode(
    model, tokenizer, context_length: int, steps: int, warmup: int, repeats: int
):
    input_ids, attention_mask = make_inputs(tokenizer, context_length)

    def prepare_cache():
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        cache = result.past_key_values
        next_token = result.logits[:, -1:].argmax(dim=-1)
        return cache, next_token

    def decode_steps(cache, next_token) -> None:
        with torch.inference_mode():
            mask = attention_mask
            for _ in range(steps):
                mask = torch.cat(
                    [mask, torch.ones((1, 1), dtype=mask.dtype, device="cuda")], dim=1
                )
                result = model(
                    input_ids=next_token,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = result.past_key_values
                next_token = result.logits[:, -1:].argmax(dim=-1)

    for _ in range(warmup):
        warmup_cache, warmup_token = prepare_cache()
        decode_steps(warmup_cache, warmup_token)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        cache, next_token = prepare_cache()
        torch.cuda.synchronize()
        samples.append(cuda_elapsed_ms(lambda: decode_steps(cache, next_token)) / steps)
    summary = summarize(samples)
    summary.update(
        {
            "initial_context_length": context_length,
            "decode_steps": steps,
            "tokens_per_second": round(1000 / summary["median_ms"], 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--decode-context", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).eval().to("cuda")

    props = torch.cuda.get_device_properties(0)
    started = time.time()
    prefill = [
        benchmark_prefill(model, tokenizer, length, args.warmup, args.repeats)
        for length in args.sequence_lengths
    ]
    decode = benchmark_decode(
        model,
        tokenizer,
        args.decode_context,
        args.decode_steps,
        args.warmup,
        args.repeats,
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "model": args.model,
        "dtype": "bfloat16",
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "duration_seconds": round(time.time() - started, 3),
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_gib": round(props.total_memory / 2**30, 3),
        },
        "prefill": prefill,
        "decode": decode,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark to {args.output}")


if __name__ == "__main__":
    main()
