#!/usr/bin/env python3
"""Kernel-level fallback for Nsight Compute (blocked: ERR_NVGPUCTRPERM).

Uses torch.profiler to get per-kernel GPU time + DRAM traffic, which gives
the same memory-bound vs compute-bound conclusion as ncu.

Usage:
  python profiling/profile_kernels.py --model-dir /workspace/models/llama-3.2-3b --num-decode 16
"""
import argparse
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--num-decode", type=int, default=16)
ap.add_argument("--prompt", default="The capital of France is")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir, gguf_file="model.gguf")
model = AutoModelForCausalLM.from_pretrained(
    args.model_dir, gguf_file="model.gguf",
    dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
)
model.eval()
dev = next(model.parameters()).device
input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(dev)

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA], profile_memory=False) as prof:
        with record_function("decode_loop"):
            past = None
            cur = input_ids
            for _ in range(args.num_decode):
                out = model(cur, past_key_values=past, use_cache=True)
                nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
                past = out.past_key_values
                cur = nxt.unsqueeze(0)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
