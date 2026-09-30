#!/usr/bin/env python3
"""Verify a HF safetensors checkpoint loads and report param count."""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/llama-3.2-3b-hf"

# Transformers 4.57.6 misclassifies local Llama configs as affected Mistral
# tokenizers. Explicit false avoids that false-positive compatibility patch.
tok = AutoTokenizer.from_pretrained(path, fix_mistral_regex=False)
print("tokenizer vocab:", len(tok))

model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
)
n = sum(p.numel() for p in model.parameters())
print(f"model params: {n/1e9:.3f} B")
print("eos_id:", tok.eos_token_id)

# smoke generation
inputs = tok("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
print("gen:", repr(tok.decode(out[0].tolist(), skip_special_tokens=True)))
