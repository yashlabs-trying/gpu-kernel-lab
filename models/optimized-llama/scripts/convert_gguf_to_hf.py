#!/usr/bin/env python3
"""Convert the fp16 GGUF model to a HuggingFace safetensors checkpoint.

Loads model.gguf via transformers (de-quantizing to fp16) and saves it back
out as config.json + model.safetensors + tokenizer, so vLLM/TensorRT can
consume it directly. No HF token required.

Usage:
  python scripts/convert_gguf_to_hf.py \
      --gguf /workspace/models/llama-3.2-3b/model.gguf \
      --out /workspace/models/llama-3.2-3b-hf
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.gguf, gguf_file="model.gguf")
    model = AutoModelForCausalLM.from_pretrained(
        args.gguf,
        gguf_file="model.gguf",
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tok.save_pretrained(args.out)
    model.save_pretrained(args.out, safe_serialization=True)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
