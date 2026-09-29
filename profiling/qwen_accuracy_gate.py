"""Large Qwen accuracy gate with operation-level drift localization.

Run on a CUDA GPU after the model is cached. The exact RMS mode validates the
harness; fast modes evaluate Triton substitutions. Only a bounded prompt subset
captures intermediate tensors, while every prompt contributes logits metrics.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "profiling"))
from accuracy_utils import StreamingTensorMetrics, deterministic_prompts, evaluate_gate
from model_ablation import install


LAYER_RE = re.compile(r"^model\.layers\.(\d+)$")
DETAIL_SUFFIXES = (
    "input_layernorm", "post_attention_layernorm", "self_attn.q_proj",
    "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_norm",
    "self_attn.k_norm", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
    "mlp.down_proj",
)


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class OperationTrace:
    def __init__(self, model):
        self.enabled = False
        self.detailed = False
        self.values = defaultdict(list)
        self.handles = []
        for name, module in model.named_modules():
            if LAYER_RE.match(name) or name in ("model.norm", "lm_head") or name.endswith(DETAIL_SUFFIXES):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def save(_module, _inputs, output):
            if not self.enabled:
                return
            if not self.detailed and not (LAYER_RE.match(name) or name in ("model.norm", "lm_head")):
                return
            tensor = first_tensor(output)
            if tensor is not None:
                self.values[name].append(tensor.detach())
        return save

    def begin(self, detailed: bool):
        self.values.clear()
        self.detailed = detailed
        self.enabled = True

    def end(self):
        self.enabled = False
        return dict(self.values)

    def close(self):
        for handle in self.handles:
            handle.remove()


def compare_traces(actual, expected, accumulators):
    missing = 0
    for name in sorted(set(actual) | set(expected)):
        a_values, e_values = actual.get(name, ()), expected.get(name, ())
        if len(a_values) != len(e_values):
            missing += abs(len(a_values) - len(e_values)) or 1
        for occurrence, (a, e) in enumerate(zip(a_values, e_values)):
            accumulators[f"{name}#{occurrence}"].update(a, e)
    return missing


def load_model(model_id, local_only):
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=local_only,
    ).eval().cuda()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--candidate", choices=("exact-rms", "fast-rms", "fast-residual-rms"), default="fast-rms")
    parser.add_argument("--prompts", type=int, default=1000)
    parser.add_argument("--trace-prompts", type=int, default=32)
    parser.add_argument("--teacher-forced-prompts", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; local CPU tests validate only the harness utilities")
    if min(args.prompts, args.decode_steps, args.max_input_tokens) <= 0:
        raise SystemExit("prompts, decode steps, and max input tokens must be positive")

    if args.prompt_file:
        prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        prompts = prompts[:args.prompts]
        if len(prompts) < args.prompts:
            raise SystemExit("prompt file contains fewer nonempty lines than --prompts")
    else:
        prompts = deterministic_prompts(args.prompts, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    baseline, candidate = load_model(args.model, args.local_files_only), load_model(args.model, args.local_files_only)
    install(candidate, "qwen_rms_exact" if args.candidate == "exact-rms" else "qwen_rms")
    if args.candidate == "fast-residual-rms":
        sys.path.insert(0, str(ROOT / "learning/12_decode_runtime"))
        from decode_runtime import install_decode_residual_norm
        install_decode_residual_norm(candidate)

    baseline_trace, candidate_trace = OperationTrace(baseline), OperationTrace(candidate)
    operation_metrics = defaultdict(StreamingTensorMetrics)
    prefill_metrics, decode_metrics = StreamingTensorMetrics(), StreamingTensorMetrics()
    prefill_matches = decode_matches = decode_tokens = missing = 0
    baseline_nll = candidate_nll = 0.0
    nll_count = 0

    try:
        for prompt_index, text in enumerate(prompts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=args.max_input_tokens)
            ids = encoded.input_ids.cuda()
            mask = encoded.attention_mask.cuda()
            trace_this = prompt_index < args.trace_prompts

            baseline_trace.begin(detailed=False) if trace_this else None
            base_out = baseline(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
            base_capture = baseline_trace.end() if trace_this else {}
            candidate_trace.begin(detailed=False) if trace_this else None
            cand_out = candidate(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
            cand_capture = candidate_trace.end() if trace_this else {}
            if trace_this:
                missing += compare_traces(cand_capture, base_capture, operation_metrics)

            base_logits, cand_logits = base_out.logits[:, -1], cand_out.logits[:, -1]
            prefill_metrics.update(cand_logits, base_logits)
            prefill_matches += int(torch.equal(cand_logits.argmax(-1), base_logits.argmax(-1)))

            if prompt_index < args.teacher_forced_prompts and ids.shape[1] > 1:
                baseline_nll += float(baseline(input_ids=ids, labels=ids, use_cache=False).loss)
                candidate_nll += float(candidate(input_ids=ids, labels=ids, use_cache=False).loss)
                nll_count += 1

            base_cache, cand_cache = base_out.past_key_values, cand_out.past_key_values
            token = base_logits.argmax(-1, keepdim=True)
            current_mask = mask
            for _step in range(args.decode_steps):
                current_mask = torch.cat((current_mask, torch.ones_like(current_mask[:, :1])), dim=1)
                baseline_trace.begin(detailed=True) if trace_this else None
                base_out = baseline(input_ids=token, attention_mask=current_mask,
                                    past_key_values=base_cache, use_cache=True, logits_to_keep=1)
                base_capture = baseline_trace.end() if trace_this else {}
                candidate_trace.begin(detailed=True) if trace_this else None
                cand_out = candidate(input_ids=token, attention_mask=current_mask,
                                     past_key_values=cand_cache, use_cache=True, logits_to_keep=1)
                cand_capture = candidate_trace.end() if trace_this else {}
                if trace_this:
                    missing += compare_traces(cand_capture, base_capture, operation_metrics)
                base_cache, cand_cache = base_out.past_key_values, cand_out.past_key_values
                base_logits, cand_logits = base_out.logits[:, -1], cand_out.logits[:, -1]
                decode_metrics.update(cand_logits, base_logits)
                decode_matches += int(torch.equal(cand_logits.argmax(-1), base_logits.argmax(-1)))
                decode_tokens += 1
                token = base_logits.argmax(-1, keepdim=True)
    finally:
        baseline_trace.close()
        candidate_trace.close()

    operations = {name: metric.as_dict() for name, metric in sorted(operation_metrics.items())}
    layer_cosines = [row["min_cosine"] for name, row in operations.items() if LAYER_RE.match(name.split("#", 1)[0])]
    minimum_layer_cosine = min(layer_cosines, default=1.0)
    mean_base_nll = baseline_nll / nll_count if nll_count else 0.0
    mean_candidate_nll = candidate_nll / nll_count if nll_count else 0.0
    all_finite = prefill_metrics.finite and decode_metrics.finite and all(x.finite for x in operation_metrics.values())
    gate = evaluate_gate(
        prefill_agreement=prefill_matches / len(prompts),
        decode_agreement=decode_matches / decode_tokens,
        minimum_layer_cosine=minimum_layer_cosine,
        nll_delta=mean_candidate_nll - mean_base_nll,
        finite=all_finite, missing_checkpoints=missing,
    )
    report = {
        "schema_version": 1,
        "model": args.model,
        "candidate": args.candidate,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        },
        "workload": {
            "prompts": len(prompts), "trace_prompts": min(args.trace_prompts, len(prompts)),
            "teacher_forced_prompts": nll_count, "decode_steps": args.decode_steps,
            "max_input_tokens": args.max_input_tokens, "seed": args.seed,
        },
        "prefill": {**prefill_metrics.as_dict(), "argmax_agreement": prefill_matches / len(prompts)},
        "decode": {**decode_metrics.as_dict(), "argmax_agreement": decode_matches / decode_tokens},
        "quality": {
            "baseline_mean_nll": mean_base_nll, "candidate_mean_nll": mean_candidate_nll,
            "nll_delta": mean_candidate_nll - mean_base_nll,
        },
        "minimum_layer_cosine": minimum_layer_cosine,
        "missing_checkpoint_values": missing,
        "operations": operations,
        "gate": gate,
        "limitations": [
            "Built-in prompts are deterministic original probes, not a standardized task benchmark.",
            "Exact-rms validates the comparison harness but is not a performance candidate.",
            "Full static-cache/quantized-path acceptance remains in rmsnorm_acceptance.py and static_kv_quality.py.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "operations"}, indent=2))
    if not gate["pass"] and not args.no_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
