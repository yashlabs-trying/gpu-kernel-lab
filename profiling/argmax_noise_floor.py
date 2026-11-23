"""Measure the argmax-agreement noise floor of stock BF16 inference.

The acceptance gate compares a candidate kernel path with the stock BF16 model
and requires >=99% argmax agreement. But stock BF16 is itself an approximation:
at positions where the top two logits are nearly tied, any change in reduction
order can flip the argmax without the model being meaningfully worse.

This tool measures that floor. It greedily generates continuations with an FP32
reference, then teacher-forces the same token sequences through:

* the FP32 reference (eager attention),
* stock BF16 with SDPA (the current gate's baseline),
* optionally stock BF16 with eager attention (a second, equally valid BF16 path).

It reports pairwise argmax agreement, how many disagreements fall on near-ties
(small FP32 top-2 margin), and mean NLL against the reference tokens. If stock
BF16 vs FP32 is itself well below 99%, a fairer gate for a BF16 kernel is:
"candidate vs FP32 agreement is no worse than stock BF16 vs FP32".

Usage (GPU):
    python profiling/argmax_noise_floor.py --output results/noise_floor.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from accuracy_utils import deterministic_prompts  # noqa: E402


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor,
                   targets: torch.Tensor, tie_margin: float) -> dict:
    """Compare two [positions, vocab] logit matrices scored on the same tokens.

    ``reference`` defines the near-tie margin (top-1 minus top-2 logit).
    """
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("expected two logit matrices of identical [positions, vocab] shape")
    if targets.shape != reference.shape[:1]:
        raise ValueError("targets must have one token per position")
    reference = reference.float()
    candidate = candidate.float()
    ref_top = reference.topk(2, dim=-1)
    margin = ref_top.values[:, 0] - ref_top.values[:, 1]
    ref_arg = ref_top.indices[:, 0]
    cand_arg = candidate.argmax(dim=-1)
    disagree = ref_arg != cand_arg
    near_tie = margin < tie_margin
    positions = int(reference.shape[0])

    def nll(logits: torch.Tensor) -> float:
        return float(torch.nn.functional.cross_entropy(logits, targets, reduction="mean"))

    return {
        "positions": positions,
        "argmax_agreement": 1.0 - float(disagree.float().mean()) if positions else 1.0,
        "disagreements": int(disagree.sum()),
        "disagreements_on_near_ties": int((disagree & near_tie).sum()),
        "near_tie_positions": int(near_tie.sum()),
        "tie_margin": tie_margin,
        "reference_nll": nll(reference),
        "candidate_nll": nll(candidate),
        "nll_delta": nll(candidate) - nll(reference),
    }


@torch.inference_mode()
def _score(model, sequences, prompt_lengths, device) -> torch.Tensor:
    rows = []
    for ids, start in zip(sequences, prompt_lengths):
        logits = model(input_ids=ids[None].to(device)).logits[0].float().cpu()
        # logits[t] predicts token t+1; keep only generated positions
        rows.append(logits[start - 1:ids.shape[0] - 1])
    return torch.cat(rows)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--tie-margin", type=float, default=0.125,
                        help="FP32 top-1 minus top-2 logit below which a position counts as a near-tie")
    parser.add_argument("--skip-bf16-eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").to(device).eval()

    sequences, prompt_lengths = [], []
    with torch.inference_mode():
        for prompt in deterministic_prompts(args.prompts):
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to(device)
            out = ref.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False)[0].cpu()
            sequences.append(out)
            prompt_lengths.append(inputs["input_ids"].shape[1])
    targets = torch.cat([s[p:] for s, p in zip(sequences, prompt_lengths)])

    fp32 = _score(ref, sequences, prompt_lengths, device)
    del ref
    if device == "cuda":
        torch.cuda.empty_cache()

    variants = {"bf16_sdpa": "sdpa"}
    if not args.skip_bf16_eager:
        variants["bf16_eager"] = "eager"
    scored = {}
    for name, attn in variants.items():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=attn).to(device).eval()
        scored[name] = _score(model, sequences, prompt_lengths, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = {
        "model": args.model,
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "prompts": args.prompts,
        "new_tokens_per_prompt": args.new_tokens,
        "comparisons": {f"{name}_vs_fp32": compare_logits(fp32, logits, targets, args.tie_margin)
                        for name, logits in scored.items()},
    }
    if "bf16_eager" in scored:
        report["comparisons"]["bf16_eager_vs_bf16_sdpa"] = compare_logits(
            scored["bf16_sdpa"], scored["bf16_eager"], targets, args.tie_margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report["comparisons"].items():
        print(f"{name:28s} agreement={result['argmax_agreement']:.4%} "
              f"disagreements={result['disagreements']} "
              f"(near-ties: {result['disagreements_on_near_ties']}) "
              f"nll_delta={result['nll_delta']:+.4f}")


if __name__ == "__main__":
    main()
