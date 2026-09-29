"""Find RMSNorm sites where Triton differs on the exact same baseline input."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learning/03_rmsnorm"))
from qwen_rmsnorm import triton_qwen_rmsnorm
sys.path.insert(0, str(ROOT / "profiling"))
from accuracy_utils import StreamingTensorMetrics, deterministic_prompts


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).eval().cuda()
    metrics = defaultdict(StreamingTensorMetrics)
    changed_elements = defaultdict(int)
    total_elements = defaultdict(int)
    calls = defaultdict(int)
    handles = []

    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen3RMSNorm":
            continue

        def compare(_module, positional, output, name=name):
            source = positional[0]
            fast = triton_qwen_rmsnorm(
                source.contiguous(), _module.weight,
                _module.variance_epsilon,
            )
            metrics[name].update(fast, output)
            changed_elements[name] += int(torch.count_nonzero(fast != output))
            total_elements[name] += output.numel()
            calls[name] += 1

        handles.append(module.register_forward_hook(compare))

    try:
        for text in deterministic_prompts(args.prompts, args.seed):
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=args.max_input_tokens)
            ids, mask = encoded.input_ids.cuda(), encoded.attention_mask.cuda()
            output = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(args.decode_steps):
                mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
                output = model(input_ids=token, attention_mask=mask, past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                token = output.logits[:, -1].argmax(-1, keepdim=True)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for name in sorted(metrics):
        row = metrics[name].as_dict()
        row.update({
            "name": name, "calls": calls[name],
            "changed_elements": changed_elements[name],
            "total_elements": total_elements[name],
            "changed_fraction": changed_elements[name] / total_elements[name],
        })
        rows.append(row)
    report = {
        "model": args.model, "gpu": torch.cuda.get_device_name(),
        "prompts": args.prompts, "decode_steps": args.decode_steps,
        "sensitive_modules": [row["name"] for row in rows if row["changed_elements"]],
        "modules": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in sorted(rows, key=lambda value: value["changed_fraction"], reverse=True):
        if row["changed_elements"]:
            print(f'{row["name"]}: changed={row["changed_fraction"]:.6%} '
                  f'max={row["max_abs"]:.6g} cosine={row["min_cosine"]:.9f}')


if __name__ == "__main__":
    main()
