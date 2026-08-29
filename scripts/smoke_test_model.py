#!/usr/bin/env python3
"""Load Qwen on CUDA and perform a short deterministic generation."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the KernelLab smoke test.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")

    messages = [{"role": "user", "content": "Reply with one short sentence about GPUs."}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cuda")

    torch.manual_seed(0)
    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = output[0, inputs.shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
