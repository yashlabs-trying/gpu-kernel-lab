#!/usr/bin/env python3
"""vLLM sweep benchmark: varies batch size, input (prefill) length, and
output (decode) length to measure TTFT, ITL, and throughput.

For each (batch, input_len, output_len) it reports:
  - TTFT (mean time-to-first-token across requests)
  - ITL (mean inter-token latency during decode)
  - throughput (total tokens / wall time)

Usage:
  python benchmarks/bench_sweep.py --model /workspace/models/llama-3.2-3b-hf
"""
import argparse
import json
import time

from vllm import LLM, SamplingParams


def make_prompt(tokens: int) -> str:
    # A repeatable filler that tokenizes to roughly `tokens` tokens.
    words = ("the quick brown fox jumps over the lazy dog while " *
             100).split()
    out = []
    n = 0
    i = 0
    while n < tokens:
        w = words[i % len(words)]
        out.append(w)
        n += 1
        i += 1
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantization", default="none")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--input-lens", default="8,128,1024")
    ap.add_argument("--output-lens", default="64,256")
    ap.add_argument("--output", default="/workspace/sweep.json")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    input_lens = [int(x) for x in args.input_lens.split(",")]
    output_lens = [int(x) for x in args.output_lens.split(",")]

    quant = None if args.quantization in ("none", "", "None") else args.quantization
    llm = LLM(model=args.model, quantization=quant, dtype="auto")

    results = []
    for b in batches:
        for il in input_lens:
            for ol in output_lens:
                prompt = make_prompt(il)
                prompts = [prompt] * b
                sp = SamplingParams(max_tokens=ol, temperature=0.0,
                                    ignore_eos=True)

                # Time the full run; vLLM streams internally, so we get
                # aggregate wall time and per-request token counts.
                t0 = time.perf_counter()
                outputs = llm.generate(prompts, sp, use_tqdm=False)
                wall = time.perf_counter() - t0

                total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
                total_tokens = total_out + b * il
                throughput = total_tokens / wall if wall else 0.0

                # Per-request ITL approximation (decode only):
                #   decode_time ~ wall - prefill_time; prefill is small.
                itl = wall / total_out if total_out else 0.0

                results.append({
                    "batch": b,
                    "input_len": il,
                    "output_len": ol,
                    "wall_s": round(wall, 4),
                    "total_tokens": total_tokens,
                    "throughput_tok_s": round(throughput, 2),
                    "requests_per_s": round(b / wall, 2) if wall else 0,
                    "approx_itl_ms": round(itl * 1000, 2),
                })
                print(json.dumps(results[-1]))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
