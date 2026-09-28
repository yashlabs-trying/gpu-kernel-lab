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
    past = None
    cur = input_ids
    for _ in range(args.num_decode):
        out = model(cur, past_key_values=past, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
        past = out.past_key_values
        cur = nxt.unsqueeze(0)
torch.cuda.synchronize()
print("done", args.num_decode, "decode steps")
