#!/usr/bin/env python3
"""vLLM offline throughput benchmark (batched inference).

Measures total throughput (tok/s) and requests/s using the synchronous LLM
class, which runs continuous batching over the prompt list internally.

Usage:
  python benchmarks/bench_vllm.py --model /workspace/models/llama-3.2-3b-hf \
      --quantization none --num-prompts 128 --max-tokens 128
"""
import argparse
import json
import random
import time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="none")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    quant = None if args.quantization in ("none", "", "None") else args.quantization
    llm = LLM(model=args.model, quantization=quant, dtype="auto")
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    prompts = [random.choice(POOL) for _ in range(args.num_prompts)]

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    result = {
        "engine": "vllm",
        "quantization": args.quantization,
        "num_prompts": args.num_prompts,
        "total_tokens": total_tokens,
        "wall_s": wall,
        "throughput_tok_s": total_tokens / wall if wall else 0,
        "requests_per_s": args.num_prompts / wall if wall else 0,
    }
    out = args.output if args.output else "/workspace/vllm_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT_FILE=" + out)


if __name__ == "__main__":
    main()
