#!/usr/bin/env python3
"""Offline vLLM throughput sweep with honest input/output accounting.

This synchronous API does not expose token arrival timestamps, so this script
does not report TTFT or ITL. Use ``bench_serving.py`` for streaming latency.
"""

from __future__ import annotations

import argparse
import json
import time

from methodology import environment_metadata, write_json

PROMPT_SEED = "the quick brown fox jumps over the lazy dog while"


def exact_prompt_ids(tokenizer, tokens: int) -> list[int]:
    seed = tokenizer(PROMPT_SEED, add_special_tokens=False).input_ids
    if not seed:
        raise ValueError("prompt seed tokenized to zero tokens")
    return (seed * ((tokens + len(seed) - 1) // len(seed)))[:tokens]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    parser.add_argument("--input-lens", default="8,128,512,2048,8192")
    parser.add_argument("--output-lens", default="32,128,256")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    concurrencies = [int(value) for value in args.concurrencies.split(",")]
    input_lens = [int(value) for value in args.input_lens.split(",")]
    output_lens = [int(value) for value in args.output_lens.split(",")]
    if min(concurrencies + input_lens + output_lens) < 1:
        parser.error("all concurrency and token lengths must be positive")

    quantization = None if args.quantization.lower() == "none" else args.quantization
    llm = LLM(
        model=args.model,
        quantization=quantization,
        dtype="float16",
        enforce_eager=args.enforce_eager,
    )
    tokenizer = llm.get_tokenizer()
    params_by_length = {
        length: SamplingParams(
            max_tokens=length,
            min_tokens=length,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        for length in output_lens
    }

    rows = []
    for concurrency in concurrencies:
        for input_len in input_lens:
            prompt_token_ids = exact_prompt_ids(tokenizer, input_len)
            prompts = [{"prompt_token_ids": prompt_token_ids}] * concurrency
            for output_len in output_lens:
                params = params_by_length[output_len]
                for _ in range(args.warmups):
                    llm.generate(prompts, params, use_tqdm=False)

                samples = []
                for repetition in range(args.repetitions):
                    start_ns = time.perf_counter_ns()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    end_ns = time.perf_counter_ns()
                    wall_s = (end_ns - start_ns) / 1_000_000_000
                    actual_input = sum(len(output.prompt_token_ids) for output in outputs)
                    actual_output = sum(
                        len(output.outputs[0].token_ids) for output in outputs
                    )
                    samples.append(
                        {
                            "repetition": repetition,
                            "start_ns": start_ns,
                            "end_ns": end_ns,
                            "wall_seconds": wall_s,
                            "input_tokens": actual_input,
                            "output_tokens": actual_output,
                            "input_tokens_per_second": actual_input / wall_s,
                            "output_tokens_per_second": actual_output / wall_s,
                            "requests_per_second": concurrency / wall_s,
                        }
                    )
                row = {
                    "concurrency": concurrency,
                    "input_tokens_per_request": input_len,
                    "output_tokens_per_request": output_len,
                    "warmups": args.warmups,
                    "repetitions": args.repetitions,
                    "raw_samples": samples,
                }
                rows.append(row)
                print(json.dumps(row))

    result = {
        "schema_version": 2,
        "benchmark": "llama_vllm_offline_throughput_sweep",
        "model": args.model,
        "dtype": "float16",
        "quantization": args.quantization,
        "cuda_graphs": not args.enforce_eager,
        "seed": args.seed,
        "sampling": {"temperature": 0.0, "ignore_eos": True},
        "methodology": {
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "latency_metrics_available": False,
            "latency_reason": "synchronous offline API has no token arrival timestamps",
            "input_and_output_throughput_reported_separately": True,
        },
        "environment": environment_metadata(),
        "measurements": rows,
    }
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
