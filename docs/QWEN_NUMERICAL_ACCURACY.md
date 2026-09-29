# Qwen numerical-accuracy repair gate

This gate prevents a fast kernel from becoming the default merely because it
passes loose tensor tolerances. It compares a fresh BF16 Transformers model with
an independently loaded candidate and requires model-level agreement.

## What is measured

The large gate records:

- prefill final-logit max/mean error, cosine, and argmax agreement;
- multi-step teacher-forced decode error and argmax agreement;
- mean teacher-forced NLL delta;
- decoder-layer boundary cosine;
- operation-level drift for RMSNorm, Q/K/V projections, Q/K normalization,
  output projection, gate/up/down projections, final norm, and LM head;
- missing or reordered checkpoints;
- NaN/Inf failures;
- GPU, compute capability, PyTorch, and Transformers versions.

Intermediate tensors are captured only for a bounded subset of prompts. Every
prompt still contributes to final-logit and decode agreement. This keeps memory
bounded while providing enough detailed traces to identify the first divergent
operation.

The built-in 1,000-prompt set is deterministic and copyright-free. A stronger
external corpus can be supplied as a plain UTF-8 file with one prompt per line.

## Acceptance requirements

| Check | Requirement |
|---|---:|
| Prefill argmax agreement | at least 99% |
| Decode argmax agreement | at least 99% |
| Minimum decoder-layer cosine | at least 0.999 |
| Absolute mean NLL delta | at most 0.01 |
| NaN or Inf | none |
| Missing/reordered checkpoints | none |

`profiling/rmsnorm_acceptance.py` separately checks 60 adversarial tensor shapes
and the complete static-cache/quantized experimental composition. Its gate now
requires both the RMS-only and full candidate to reach 99% argmax agreement.

## GPU execution order

Activate the RunPod environment and make sure `Qwen/Qwen3-0.6B` is already in
the persistent Hugging Face cache.

### 1. Validate the harness with the exact implementation

```bash
python profiling/qwen_accuracy_gate.py \
  --candidate exact-rms \
  --prompts 1000 \
  --trace-prompts 32 \
  --teacher-forced-prompts 64 \
  --decode-steps 4 \
  --local-files-only \
  --output results/qwen_accuracy/exact_rms.json
```

This must pass. Failure indicates a comparison or environment problem, not an
optimization problem.

### 2. Measure standalone fast RMSNorm

```bash
python profiling/qwen_accuracy_gate.py \
  --candidate fast-rms \
  --prompts 1000 \
  --trace-prompts 32 \
  --teacher-forced-prompts 64 \
  --decode-steps 4 \
  --local-files-only \
  --output results/qwen_accuracy/fast_rms.json
```

Inspect the earliest operation whose cosine or maximum error departs from the
exact run. Do not proceed to larger fusion until this result is understood.

### 3. Measure cross-layer residual/RMSNorm fusion

```bash
python profiling/qwen_accuracy_gate.py \
  --candidate fast-residual-rms \
  --prompts 1000 \
  --trace-prompts 32 \
  --teacher-forced-prompts 64 \
  --decode-steps 4 \
  --local-files-only \
  --output results/qwen_accuracy/fast_residual_rms.json
```

If the fast kernel fails first at RMSNorm, test the precision-preserving Triton
math path before falling back to eager PyTorch:

```bash
python profiling/qwen_accuracy_gate.py \
  --candidate precise-rms \
  --prompts 1000 \
  --trace-prompts 32 \
  --teacher-forced-prompts 64 \
  --decode-steps 4 \
  --local-files-only \
  --output results/qwen_accuracy/precise_rms.json
```

This variant retains the Triton kernel and FP32 reduction but computes inverse
RMS with a square root followed by division rather than approximate reciprocal
square root. It is accepted only if both the accuracy and latency gates pass.

### 4. Run adversarial and full-composition acceptance

```bash
python profiling/rmsnorm_acceptance.py \
  --prompts 1000 \
  --output results/qwen_accuracy/full_acceptance.json

python profiling/static_kv_quality.py \
  --hybrid --split-attention --residual-norm \
  --output results/qwen_accuracy/static_kv_quality.json
```

The commands intentionally exit with status 2 when an acceptance gate fails.
Use `--no-fail` only on `qwen_accuracy_gate.py` during diagnosis, never for a
release decision.

## How to locate a failure

Read the `operations` mapping in the JSON report in execution order:

```text
input_layernorm
  -> q/k/v projection
  -> q_norm / k_norm
  -> attention output projection
  -> post_attention_layernorm
  -> gate/up projection
  -> down projection
  -> decoder-layer output
  -> final norm
  -> LM head
```

If the exact run passes but fast RMSNorm fails first at an RMSNorm checkpoint,
test reduction ordering, reciprocal-square-root behavior, and BF16 casting. If
RMSNorm passes but the residual version fails at the following layer boundary,
test residual-add rounding and the cross-layer handoff. If only the full
composition fails, isolate static attention and quantized projections separately.

To identify RMSNorm modules that differ while receiving the exact same baseline
input, run:

```bash
python profiling/qwen_rms_sensitivity.py \
  --prompts 64 --decode-steps 4 --local-files-only \
  --output results/qwen_accuracy/rms_sensitivity.json
```

This distinguishes local kernel rounding from error propagated by an earlier
layer. It reports changed-element fractions for each named norm site.

## CPU validation

The GPU model cannot be certified on CPU, but the deterministic corpus, streaming
metrics, NaN handling, and threshold logic are CPU-tested:

```bash
pytest -q profiling/test_accuracy_utils.py
```
