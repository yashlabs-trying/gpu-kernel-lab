#!/usr/bin/env python3
"""PyTorch fp16 baseline for Llama-3.2-3B.

Loads the model (GGUF staged as fp16) into PyTorch and does a manual
greedy decode loop with per-token timing so we can measure:

  TTFT       = time-to-first-token (prefill latency)
  ITL        = inter-token latency (decode time per token)
  throughput = total tokens / total wall time

Usage:
  python benchmarks/bench_torch_baseline.py --model-dir /workspace/models/llama-3.2-3b \
      --max-new-tokens 128
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, gguf_file="model.gguf")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        gguf_file="model.gguf",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def run(prompt: str, tokenizer, model, max_new_tokens: int):
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    ttft = 0.0
    itls = []
    generated = []

    t0 = time.perf_counter()
    past_key_values = None
    cur_ids = input_ids

    with torch.no_grad():
        for step in range(max_new_tokens):
            out = model(cur_ids, past_key_values=past_key_values, use_cache=True)
            logits = out.logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1)
            generated.append(next_id.item())
            past_key_values = out.past_key_values
            cur_ids = next_id.unsqueeze(0)

            if step == 0:
                ttft = time.perf_counter() - t0
            else:
                itls.append(time.perf_counter() - last)
            last = time.perf_counter()

            if next_id.item() == tokenizer.eos_token_id:
                break

    total_time = time.perf_counter() - t0
    n_tokens = len(generated)
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "ttft_s": ttft,
        "num_tokens": n_tokens,
        "total_time_s": total_time,
        "mean_itl_s": sum(itls) / len(itls) if itls else 0.0,
        "decode_tok_s": (n_tokens - 1) / sum(itls) if itls else 0.0,
        "throughput_tok_s": n_tokens / total_time if total_time else 0.0,
        "output": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt", default="What is the capital of France?")
    args = ap.parse_args()

    tokenizer, model = load(args.model_dir)
    result = run(args.prompt, tokenizer, model, args.max_new_tokens)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
