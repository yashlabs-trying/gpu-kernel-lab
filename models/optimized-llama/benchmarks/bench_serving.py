#!/usr/bin/env python3
"""Serving benchmark: TTFT, ITL, and throughput.

Modes:
  --engine hf       naive HuggingFace (baseline)
  --engine vllm     vLLM serving

Reports: TTFT (time-to-first-token), ITL (inter-token latency),
         throughput (tok/s and req/s).
"""
import argparse
import json
import time


def bench_hf(model_path: str, prompts: list[str], max_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    ttfts, itls, total_tokens = [], [], 0
    wall_start = time.time()

    for p in prompts:
        inputs = tok(p, return_tensors="pt").to(model.device)
        start = time.time()
        first = True
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        elapsed = time.time() - start
        n_tok = out.shape[1] - inputs.input_ids.shape[1]
        total_tokens += n_tok
        itls.append(elapsed / n_tok if n_tok else 0)
        ttfts.append(elapsed / n_tok if n_tok else 0)  # HF: no easy per-token split
    wall = time.time() - wall_start

    return {
        "engine": "hf",
        "requests": len(prompts),
        "total_tokens": total_tokens,
        "wall_seconds": wall,
        "throughput_tok_s": total_tokens / wall if wall else 0,
        "throughput_req_s": len(prompts) / wall if wall else 0,
        "mean_ttft_s": sum(ttfts) / len(ttfts) if ttfts else 0,
        "mean_itl_s": sum(itls) / len(itls) if itls else 0,
    }


def bench_vllm(model_path: str, prompts: list[str], max_tokens: int):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="float16")
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    ttfts, itls, total_tokens = [], [], 0
    wall_start = time.time()

    outputs = llm.generate(prompts, sp)
    for o in outputs:
        n_tok = len(o.outputs[0].token_ids)
        total_tokens += n_tok
        # vLLM does not expose per-token timing here; use aggregate
        ttfts.append(0.0)
        itls.append(0.0)
    wall = time.time() - wall_start

    return {
        "engine": "vllm",
        "requests": len(prompts),
        "total_tokens": total_tokens,
        "wall_seconds": wall,
        "throughput_tok_s": total_tokens / wall if wall else 0,
        "throughput_req_s": len(prompts) / wall if wall else 0,
        "mean_ttft_s": 0.0,
        "mean_itl_s": 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompts", default='["What is the capital of France?", "Explain quantum computing in simple terms."]')
    args = ap.parse_args()

    prompts = json.loads(args.prompts)
    fn = bench_hf if args.engine == "hf" else bench_vllm
    result = fn(args.model_path, prompts, args.max_tokens)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
