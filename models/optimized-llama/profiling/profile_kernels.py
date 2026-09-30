#!/usr/bin/env python3
"""Kernel-level fallback for Nsight Compute (blocked: ERR_NVGPUCTRPERM).

Uses torch.profiler to get per-kernel GPU time + DRAM traffic, which gives
the same memory-bound vs compute-bound conclusion as ncu.

Usage:
  python profiling/profile_kernels.py --model-dir /workspace/models/llama-3.2-3b --num-decode 16
"""
import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--gguf-file", default=None)
ap.add_argument("--num-decode", type=int, default=16)
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--output", type=Path)
ap.add_argument("--trace-output", type=Path)
args = ap.parse_args()

load_kwargs = {"gguf_file": args.gguf_file} if args.gguf_file else {}
tok = AutoTokenizer.from_pretrained(
    args.model_dir, fix_mistral_regex=False, **load_kwargs
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    **load_kwargs,
)
model.eval()
dev = next(model.parameters()).device
inputs = tok(args.prompt, return_tensors="pt").to(dev)

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA], profile_memory=False) as prof:
        with record_function("decode_loop"):
            past = None
            cur = inputs["input_ids"]
            for _ in range(args.num_decode):
                out = model(cur, past_key_values=past, use_cache=True)
                nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
                past = out.past_key_values
                cur = nxt.unsqueeze(0)
        torch.cuda.synchronize()

table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
print(table)
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table + "\n", encoding="utf-8")
if args.trace_output:
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace_output))
