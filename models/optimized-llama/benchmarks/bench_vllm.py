#!/usr/bin/env python3
"""Focused vLLM offline throughput benchmark.

No TTFT/ITL is reported because the synchronous offline API has no streamed
token timestamps. Use bench_serving.py for latency metrics.
"""

from __future__ import annotations

import argparse
import random
import time

from methodology import environment_metadata, write_json

POOL = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main causes of climate change?",
    "Describe how photosynthesis works.",
    "What is the difference between RAM and ROM?",
    "List five healthy breakfast ideas.",
    "How does a blockchain work?",
    "What is machine learning?",
    "Explain the water cycle.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    prompts = [rng.choice(POOL) for _ in range(args.num_prompts)]
    quantization = None if args.quantization.lower() == "none" else args.quantization
    llm = LLM(
        model=args.model,
        quantization=quantization,
        dtype="float16",
        enforce_eager=args.enforce_eager,
    )
    params = SamplingParams(
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
    )

    for _ in range(args.warmups):
        llm.generate(prompts, params, use_tqdm=False)

    samples = []
    for repetition in range(args.repetitions):
        start_ns = time.perf_counter_ns()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        end_ns = time.perf_counter_ns()
        wall_s = (end_ns - start_ns) / 1_000_000_000
        input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        samples.append(
            {
                "repetition": repetition,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "wall_seconds": wall_s,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_tokens_per_second": input_tokens / wall_s,
                "output_tokens_per_second": output_tokens / wall_s,
                "requests_per_second": args.num_prompts / wall_s,
            }
        )

    result = {
        "schema_version": 2,
        "benchmark": "llama_vllm_offline_throughput",
        "engine": "vllm-offline",
        "model": args.model,
        "quantization": args.quantization,
        "dtype": "float16",
        "concurrency": args.num_prompts,
        "cuda_graphs": not args.enforce_eager,
        "seed": args.seed,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "output_tokens": args.max_tokens,
        },
        "methodology": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "latency_metrics_available": False,
            "latency_reason": "synchronous offline API has no token arrival timestamps",
        },
        "environment": environment_metadata(),
        "raw_samples": samples,
    }
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
