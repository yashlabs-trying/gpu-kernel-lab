#!/usr/bin/env python3
"""Capture deterministic generated tokens and per-step logits on a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--quantization",
        choices=("none", "bitsandbytes-int8", "bitsandbytes-int4"),
        default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, fix_mistral_regex=False
    )
    model_kwargs = {"dtype": dtype, "device_map": "auto"}
    if args.quantization == "bitsandbytes-int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "bitsandbytes-int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    device = next(model.parameters()).device
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    current = prompt_ids
    cache = None
    logits: list[np.ndarray] = []
    generated: list[int] = []
    layer_states: list[list[np.ndarray]] | None = None
    with torch.inference_mode():
        for _ in range(args.output_tokens):
            output = model(
                current,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            step_logits = output.logits[:, -1, :].float()
            next_id = step_logits.argmax(dim=-1)
            logits.append(step_logits.cpu().numpy()[0])
            generated.append(int(next_id.item()))
            hidden_states = output.hidden_states or ()
            if layer_states is None:
                layer_states = [[] for _ in hidden_states]
            for index, hidden in enumerate(hidden_states):
                layer_states[index].append(hidden[:, -1, :].float().cpu().numpy()[0])
            cache = output.past_key_values
            current = next_id.unsqueeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "logits": np.stack(logits),
        "generated_token_ids": np.asarray(generated, dtype=np.int64),
        "prompt_token_ids": prompt_ids.cpu().numpy()[0],
        "metadata": np.asarray(
            json.dumps(
                {
                    "model": args.model,
                    "dtype": args.dtype,
                    "quantization": args.quantization,
                    "seed": args.seed,
                }
            )
        ),
    }
    for index, values in enumerate(layer_states or []):
        arrays[f"layer_{index:02d}"] = np.stack(values)
    np.savez_compressed(args.output, **arrays)
    print(args.output)


if __name__ == "__main__":
    main()
