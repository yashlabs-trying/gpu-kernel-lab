#!/usr/bin/env python3
"""Capture logits, tokens, and hidden states for the full quality corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def capture_prompt(model, tokenizer, prompt: str, output_tokens: int) -> dict[str, object]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    current = prompt_ids
    cache = None
    logits = []
    generated = []
    layer_states = None
    with torch.inference_mode():
        for _ in range(output_tokens):
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
    return {
        "logits": np.stack(logits),
        "generated": np.asarray(generated, dtype=np.int64),
        "prompt_ids": prompt_ids.cpu().numpy()[0],
        "layers": [np.stack(values) for values in (layer_states or [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=16)
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
    if args.output_tokens < 1:
        parser.error("--output-tokens must be positive")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        parser.error("--prompts must contain a non-empty JSON list")
    invalid_prompt = any(
        not isinstance(item, dict) or not item.get("id") or not item.get("text")
        for item in prompts
    )
    if invalid_prompt:
        parser.error("every prompt must contain non-empty id and text fields")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=False)
    model_kwargs = {"dtype": dtype, "device_map": "auto"}
    if args.quantization == "bitsandbytes-int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantization == "bitsandbytes-int4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()

    captures = [
        capture_prompt(model, tokenizer, item["text"], args.output_tokens)
        for item in prompts
    ]
    prompt_lengths = [len(capture["prompt_ids"]) for capture in captures]
    flat_prompt_ids = np.concatenate([capture["prompt_ids"] for capture in captures])
    prompt_offsets = np.cumsum([0, *prompt_lengths], dtype=np.int64)
    arrays = {
        "logits": np.stack([capture["logits"] for capture in captures]),
        "generated_token_ids": np.stack(
            [capture["generated"] for capture in captures]
        ),
        "prompt_token_ids_flat": flat_prompt_ids,
        "prompt_token_offsets": prompt_offsets,
        "metadata": np.asarray(
            json.dumps(
                {
                    "model": args.model,
                    "dtype": args.dtype,
                    "quantization": args.quantization,
                    "seed": args.seed,
                    "output_tokens": args.output_tokens,
                    "prompts": prompts,
                }
            )
        ),
    }
    layer_count = len(captures[0]["layers"])
    for index in range(layer_count):
        arrays[f"layer_{index:02d}"] = np.stack(
            [capture["layers"][index] for capture in captures]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(args.output)


if __name__ == "__main__":
    main()
