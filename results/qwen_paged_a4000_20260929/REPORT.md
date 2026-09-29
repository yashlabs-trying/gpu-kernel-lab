# Qwen Phase 3/4: Ampere validation and architecture policy

Date: 2026-09-29  
Model: `Qwen/Qwen3-0.6B`  
GPU: NVIDIA RTX A4000, 16 GiB, SM86  
Software: PyTorch 2.8.0+cu128, Triton 3.4.0, BF16

## Decision

The production policy enables the lossless fused paged-KV gather on SM86. It
keeps native-GQA, projection fusion, and direct paged attention opt-in because
they either regress some context lengths or fail the numerical gate.

Unknown GPU architectures use the exact reference fallback. SM86 and SM89 have
separate checked-in profiles; kernel selection is based on both GPU capability
and the exact model signature.

## Accuracy gate

The accepted fused-gather path was compared with the Transformers/PyTorch BF16
reference on 256 independent prompts and four decode decisions per prompt.

| Check | Result |
|---|---:|
| Prefill argmax agreement | **100.000%** (256/256) |
| Decode argmax agreement | **100.000%** (1,024/1,024) |
| Maximum absolute logit difference | **0.0** |
| Mean absolute logit difference | **0.0** |
| Minimum cosine similarity | **0.99999976** |
| Finite outputs | pass |
| KV blocks reclaimed | pass |

The best selective projection experiment (QKV fusion in layers 24, 26, and 27,
plus MLP fusion) passed the 99% release threshold, but was not selected as the
global default:

| Check | Result |
|---|---:|
| Prefill agreement | 100.000% |
| Decode agreement | 99.5117% (1,019/1,024) |
| Maximum absolute logit difference | 0.25 |
| Minimum cosine similarity | 0.9997078 |

Fusing the seven lowest-drift QKV layers produced only 98.8281% agreement and
failed. Layer-by-layer calibration therefore cannot be composed blindly.

## Stable-shape decode benchmark

Each row uses 16 warm-up and 100 measured one-token decode iterations. CUDA
events measure the full executor decode call. Fixed-width block tables prevent
new shapes from compiling at page boundaries.

| Context | Reference median | Accepted median | Reduction | Reference p99 | Accepted p99 |
|---:|---:|---:|---:|---:|---:|
| 128 | 33.519 ms | **29.086 ms** | **13.23%** | 44.463 ms | **34.021 ms** |
| 512 | 33.590 ms | **28.946 ms** | **13.83%** | 35.580 ms | **31.275 ms** |
| 2,048 | 33.534 ms | **29.230 ms** | **12.84%** | 46.701 ms | **31.379 ms** |
| 4,096 | 34.174 ms | **28.892 ms** | **15.46%** | 42.811 ms | **31.240 ms** |

These are improvements over this report's matched reference paged executor, not
over the earlier static CUDA-Graph experiment and not server throughput.

The selective three-layer QKV/MLP candidate reached 15.10%, 15.18%, 11.31%, and
11.07% reduction at the same contexts. It was slightly faster for short context
but slower than fused gather alone at 2,048 and 4,096, so the production policy
rejects it. The all-layer projection candidate reached at most 19.05% but failed
quality. The requested consistent 20% additional reduction was therefore **not
met** on this GPU.

## What changed

- Fused page lookup, K/V gather, and GQA expansion without changing arithmetic.
- Added native-GQA and combined QKV/gate-up candidates as gated experiments.
- Added per-layer QKV calibration and composition tests.
- Added fixed-width KV metadata for stable CUDA-Graph/benchmark shapes.
- Added model-signature and compute-capability policy selection.
- Made uncalibrated architectures fall back to the exact path.

## Reproduce

```bash
python -m pytest \
  serving/test_serving.py \
  serving/test_kernel_policy.py \
  serving/test_paged_attention.py -q

python profiling/qwen_paged_executor_quality.py \
  --prompts 256 --decode-steps 4 --local-files-only \
  --output results/qwen_paged_a4000_20260929/fused_gather_256.json

python profiling/qwen_paged_executor_benchmark.py \
  --contexts 128 512 2048 4096 --warmup 16 --steps 100 \
  --fused-qkv-layers 24 26 27 --local-files-only \
  --output results/qwen_paged_a4000_20260929/final_benchmark.json
```

The raw JSON files beside this report contain the accepted quality run, the
selective-fusion quality run, the failed seven-layer composition, and all timing
samples summarized above.
