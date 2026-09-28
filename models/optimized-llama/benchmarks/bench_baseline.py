#!/usr/bin/env python3
"""Clean baseline: fixed 128-token decode + pynvml util sampling.

Reports TTFT, mean ITL, steady-state decode tok/s, and GPU util/bandwidth
evidence to prove the memory-bound conclusion.
"""
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--num-decode", type=int, default=128)
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


def sample_util(handle):
    u = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return u.gpu, u.memory


handle = pynvml.nvmlDeviceGetHandleByIndex(0) if HAS_NVML else None

ttft = 0.0
itls = []
samples = []

with torch.no_grad():
    past = None
    cur = input_ids
    t0 = time.perf_counter()
    for i in range(args.num_decode):
        out = model(cur, past_key_values=past, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
        past = out.past_key_values
        cur = nxt.unsqueeze(0)
        if i == 0:
            ttft = time.perf_counter() - t0
        else:
            itls.append(time.perf_counter() - last)
        last = time.perf_counter()
        if HAS_NVML and i % 8 == 0:
            samples.append(sample_util(handle))
total = time.perf_counter() - t0

decode_tok_s = (args.num_decode - 1) / sum(itls) if itls else 0.0
mean_itl = sum(itls) / len(itls) if itls else 0.0
gpu_util = sum(s[0] for s in samples) / len(samples) if samples else 0.0
mem_util = sum(s[1] for s in samples) / len(samples) if samples else 0.0

print("=== BASELINE (PyTorch fp16, Llama-3.2-3B, A40) ===")
print(f"TTFT (prefill)        : {ttft*1000:.1f} ms")
print(f"mean ITL (decode)     : {mean_itl*1000:.2f} ms")
print(f"decode tok/s          : {decode_tok_s:.2f}")
print(f"total tok/s (e2e)     : {args.num_decode/total:.2f}")
print(f"avg GPU util (SM)     : {gpu_util:.1f}%")
print(f"avg mem util          : {mem_util:.1f}%")
print(f"model fp16 size       : ~6.4 GB")
print(f"effective BW = tok/s x 6.4GB = {decode_tok_s*6.4:.0f} GB/s "
      f"of ~696 GB/s peak")
