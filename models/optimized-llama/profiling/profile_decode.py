#!/usr/bin/env python3
"""Minimal profiling target: load model, prefill, decode N tokens.

Kept lean so Nsight Compute / Nsight Systems capture a focused window.
Usage:
  python profiling/profile_decode.py --model-dir /workspace/models/llama-3.2-3b --num-decode 16
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--gguf-file", default=None)
ap.add_argument("--num-decode", type=int, default=16)
ap.add_argument("--prompt", default="The capital of France is")
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
    past = None
    cur = inputs["input_ids"]
    for _ in range(args.num_decode):
        out = model(cur, past_key_values=past, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
        past = out.past_key_values
        cur = nxt.unsqueeze(0)
torch.cuda.synchronize()
print("done", args.num_decode, "decode steps")
