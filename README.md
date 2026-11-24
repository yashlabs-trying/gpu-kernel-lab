# GPU Kernel Lab

[![CPU tests](https://github.com/yashlabs-trying/gpu-kernel-lab/actions/workflows/cpu-tests.yml/badge.svg)](https://github.com/yashlabs-trying/gpu-kernel-lab/actions/workflows/cpu-tests.yml)
![Triton](https://img.shields.io/badge/Triton-kernels-orange)
![CUDA](https://img.shields.io/badge/CUDA-Graphs-76B900)
![GPUs](https://img.shields.io/badge/measured%20on-RTX%203090%20%7C%20A4000%20%7C%20RTX%202000%20Ada-555)
![License](https://img.shields.io/badge/license-MIT-green)

Custom Triton/CUDA kernels, a static CUDA-Graph decode runtime and a paged-KV
serving engine for [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B),
measured end to end on RTX 3090, A4000 and RTX 2000 Ada GPUs.

> **Rule of the lab:** a kernel ships only if it passes the numerical gate
> (≥99% argmax agreement with the BF16 reference) **and** improves the complete
> workload. A faster microbenchmark alone is not a model speedup.

No external optimized-kernel repository was copied. PyTorch, Triton,
Transformers, cuBLAS/CUTLASS dispatch and PyTorch SDPA/FlashAttention are
dependencies and reference implementations.

## Results at a glance

**Accepted** — passes the numerical gate, enabled by default:

| Change | Result | GPU | Report |
|---|---|---|---|
| Decode-first scheduler, stable metadata shapes, no host syncs, adaptive prefill quanta | **+13.53%** concurrent throughput, ITL p50 **−20.85%**, ITL p99 **−25.80%**; 64-request run with zero failures and zero leaked KV blocks | A4000 (SM86) | [report](results/qwen_phase56_a4000/REPORT.md) |
| Lossless paged-gather decode policy | Median paged-decode latency **−12.84% to −15.46%** across contexts 128–4,096, 100% argmax agreement, zero logit difference | A4000 (SM86) | [report](results/qwen_paged_a4000_20260929/REPORT.md) |
| Paged-KV serving executor | 256/256 prefill and 1,024/1,024 decode decisions identical to the baseline | RTX 2000 Ada (SM89) | [report](results/qwen_paged_ada_20260929/REPORT.md) |
| Exact-shape prefill dispatcher | TTFT p50 −1.02%, p99 −3.43% (only the `q_proj` winner survived end-to-end testing) | RTX 3090 | [report](results/final_newpod_20260904/REPORT.md) |

**Experimental** — much faster, opt-in because it misses the 99% gate:

| Path | Result | Argmax agreement |
|---|---|---:|
| Static CUDA-Graph decode (fused RMSNorm/RoPE/KV-write, split-KV attention, W8A16 gate/up + SwiGLU, fixed-address graph replay) | Decode ITL **22.07 → 2.25 ms (9.8×)**, kernels/token **693 → 258**, RTX 3090, context 2,048 | 96.48% |
| Direct Triton paged attention | **1.74–5.37×** faster than gather + SDPA in isolation, contexts 128–4,096 | 97.66% |

The 9.8× figure is `1000 / ITL` for a serial GPU-only decode loop; it excludes
tokenization, networking, scheduling and sampling and is not a server
throughput claim. Static decode by context (RTX 3090, BF16, batch 1):

| Context | Dynamic ITL | Graph ITL | Dynamic token/s | Graph token/s |
|---:|---:|---:|---:|---:|
| 128 | 21.906 ms | 2.318 ms | 45.65 | 431.46 |
| 512 | 21.958 ms | 2.321 ms | 45.54 | 430.92 |
| 2,048 | 22.073 ms | **2.253 ms** | 45.30 | **443.90** |
| 4,096 | 22.021 ms | 2.599 ms | 45.41 | 384.70 |


### Is the 99% gate measuring the kernels or BF16 noise?

Even the RMSNorm-fusion-only path, which changes nothing but reduction order,
reaches 98.05% agreement. That suggests part of the gap is BF16 rounding at
near-tied logits rather than kernel error. [`profiling/argmax_noise_floor.py`](profiling/argmax_noise_floor.py)
measures the floor directly: it scores the stock BF16 model against an FP32
reference on the same teacher-forced tokens, and reports how many disagreements
are near-ties. If stock BF16 itself lands near 96–98% against FP32, the fair gate
for a BF16 kernel is "no worse than stock BF16 against FP32", and the
experimental decode path should be re-evaluated against that bar.

```bash
python profiling/argmax_noise_floor.py --output results/noise_floor.json
```

## Install and CLI

The repository is now an installable package with a unified model registry and
commands for setup, serving, benchmarking, and result validation:

```bash
python -m pip install -e ".[cpu,llama]"
kernellab models
kernellab doctor
kernellab download llama-3.2-3b
kernellab serve llama-3.2-3b
kernellab benchmark llama-3.2-3b --engine pytorch --output results/llama-pytorch.json
kernellab validate-result results/llama-pytorch.json
```

See the [architecture](docs/ARCHITECTURE.md), [Llama deployment guide](docs/LLAMA_DEPLOYMENT.md),
and [reproducibility matrix](docs/REPRODUCIBILITY_MATRIX.md). GPU-dependent
claims remain pending until their schema-v2 artifacts exist.

## Models

| Model | Optimization focus | Status |
|---|---|---|
| [Qwen3-0.6B](models/qwen3-0.6b/README.md) | Custom Triton/CUDA kernels, static decode, CUDA Graphs, serving control plane | Exact production path; faster experimental paths gated |
| [Llama-3.2-3B](models/optimized-llama/README.md) | vLLM serving, matched benchmarks, profiling, kernel work | A40 baseline and vLLM sweep measured; fresh validation pending |

The Llama source is pinned at upstream commit `80087ed` and included directly,
so a normal clone receives it without submodule commands. See its
[provenance record](models/optimized-llama/UPSTREAM.md) and the
[portfolio roadmap](models/README.md).

## What we built

### Fundamental kernels

- **Vector addition:** arbitrary-size masks, coalesced access, block/warp sweeps,
  effective bandwidth, and add+SiLU fusion.
- **SwiGLU:** standalone activation and fused gate/up projection + SwiGLU, with
  separate tiled-prefill and M=1 decode strategies.
- **RMSNorm:** FP32 reduction, Qwen-exact cast ordering, residual fusion, and
  adversarial-scale validation.
- **RoPE:** Q/K pair rotation, coalesced layouts, sin/cos reuse, and joint Q/K
  processing.
- **Softmax:** stable FP32 max/sum reductions, load-once rows, masking, and
  register-pressure experiments.
- **GEMM/GEMV:** tiled GEMM, shape buckets, exact-shape prefill dispatch, Split-K,
  and warp-reduced decode GEMV.
- **Attention:** educational online tiled attention and GQA-aware split-KV decode
  attention. Optimized SDPA remains the prefill winner.

Every chapter under [`learning/`](learning/) contains code, tests, benchmarks,
and explanations of mapping, coalescing, precision, reductions, register
pressure, fusion, and trade-offs.

### Prefill integration

- Measured actual Qwen projection and MLP shapes.
- Built an exact-shape dispatcher with a PyTorch fallback.
- Tested Q, K/V, gate/up, and down-projection micro-winners end to end.
- Rejected candidates that won alone but increased complete-model TTFT.
- Retained only `(q_proj, M=2048, K=1024, N=2048)` in the accepted whitelist.
- Kept PyTorch SDPA/FlashAttention because the custom teaching kernel loses.

This is why accepted TTFT improvement is modest: results reflect integrated
performance, not a sum of unrelated microbenchmark gains.

### Decode integration

The experimental decode pipeline is:

```text
fixed token/position buffers
  -> phase-specific projections
  -> Q/K RMSNorm + RoPE + direct K/V cache and mask write
  -> GQA-aware split-KV online-softmax attention
  -> attention residual + post-attention RMSNorm
  -> W8A16 dual gate/up GEMV + SwiGLU
  -> MLP residual + next-layer RMSNorm across 27 boundaries
  -> LM-head GEMV + argmax + token/position update
  -> fixed-address CUDA Graph replay
```

This removed dynamic cache concatenation, old-cache copying, GQA expansion, and
many intermediate tensors. Attention residual/RMSNorm is fused per layer;
MLP residual/next-layer RMSNorm is fused across 27 layer boundaries. Kernel
launches fell from the early 1,624/token baseline to 258/token.

### Quantization experiments

- W4A16 decode GEMV and packed INT4 LM-head/local top-k prototypes.
- Per-output-channel weight quantization.
- W8A16 dual gate/up projection with fused SwiGLU.
- Selective BF16 protection for sensitive layers and projections.
- Teacher-forced NLL and argmax comparisons with the dynamic BF16 reference.

Quantization is not enabled universally. Some A/B paths retain dense weights, so
whole-model memory savings are not claimed unless the dense copy is removed.

### Serving system

[`serving/`](serving/) provides:

- Fixed-block paged-KV allocation with atomic reservation and immediate reuse.
- Continuous batching, admission control, priorities, and decode-first scheduling.
- Round-robin chunked prefill, cancellation, timeouts, and failure recovery.
- CUDA Graph bucket selection for common batch/sequence shapes.
- Temperature, top-k, top-p, repetition penalty, stop tokens, deterministic
  seeds, and optional log probabilities.
- OpenAI-compatible completion/chat JSON and streaming endpoints.
- Health, model, metrics, and cancellation endpoints.
- TTFT/ITL percentiles, throughput, request counts, and KV usage metrics.

The 5,000-request control-plane stress test recorded 4,864 completions and 136
intentional cancellations, with zero failures, timeouts, rejections, or leaked
KV blocks. All 60,000 blocks were reclaimed. Its 197.81 requests/s rate measures
the control plane with a fake executor—not GPU model throughput. Full details are
in the [serving report](results/final_serving_20260904/REPORT.md). Separately, a
real four-request GPU smoke matched all 16 greedy Qwen baseline tokens and
reclaimed every physical KV block; larger production traffic measurements are
still required.

## Correctness and profiling

- **476** repository tests passed in the final GPU environment (80 run on CPU in CI).
- **10** serving/API tests passed.
- 60 adversarial RMSNorm configurations covered batches, contexts, and scales.
- 256 deterministic prompts were checked with per-layer drift localization.
- A separate 247-token teacher-forced quality probe was run.
- Eager/graph replay verified tokens, positions, and fixed buffer addresses.
- Nsight Systems measured kernel structure and timeline utilization.

RMS/fusion-only quality reached 98.05% argmax agreement and minimum per-layer
cosine 0.999726. The complete experimental path reached 96.48% agreement, so it
remains opt-in. Nsight Compute counters were blocked by RunPod's
`ERR_NVGPUCTRPERM`; occupancy, DRAM bandwidth, warp stalls, and counter-derived
rooflines are therefore not invented.

## Repository map

```text
baseline/     reference implementations
learning/     kernels, tests, benchmarks, and teaching notes
models/       self-contained optimized LLM projects (Qwen3-0.6B, Llama-3.2-3B)
profiling/    workloads, ablations, quality gates, and Nsight helpers
serving/      paged KV, scheduler, sampling, metrics, engine, and API
scripts/      RunPod setup, model download, activation, and smoke tests
docs/         roadmap and optimization plans
results/      raw measurements, traces, logs, and acceptance reports
```

Recommended reading:

1. [Experiment roadmap](docs/EXPERIMENTS.md)
2. [Vector addition](learning/01_vector_add/README.md)
3. [Fused projection + SwiGLU](learning/02_swiglu/FUSED_PROJECTION.md)
4. [Qwen RMSNorm](learning/03_rmsnorm/QWEN_RMSNORM.md)
5. [GEMM and exact-shape dispatch](learning/06_gemm/README.md)
6. [Decode kernels](learning/08_decode/README.md)
7. [Static KV](learning/11_static_kv/README.md)
8. [Serving engine](serving/README.md)
9. [Final profiling report](results/final_newpod_20260904/REPORT.md)
10. [Qwen numerical-accuracy gate](docs/QWEN_NUMERICAL_ACCURACY.md)

## Reproduce on RunPod

```bash
git clone https://github.com/yashlabs-trying/gpu-kernel-lab.git
cd gpu-kernel-lab
bash scripts/setup_runpod.sh
source .venv/bin/activate
python scripts/check_environment.py
python scripts/download_model.py
python scripts/smoke_test_model.py
pytest -q
```

Key profiling commands:

```bash
python profiling/model_workload.py --output results/baseline.json
python profiling/rmsnorm_acceptance.py --help
python profiling/static_kv_benchmark.py --context 2048 --hybrid --split-attention
python profiling/decode_graph_check.py
python profiling/serving_stress.py --requests 5000 --output results/stress.json
```

Use `--help` and the commands in individual reports for exact options. Keep model
weights and large profiler artifacts on the persistent volume.

## Production boundary

Completed for the single-GPU serving core:

1. Numerical acceptance above 99%, including an exact production path.
2. Capability-based SM86/SM89 profiles with an exact unknown-GPU fallback.
3. Real concurrent GPU p50/p90/p99 measurement and complete KV reclamation.
4. A no-weights deployment preflight covering model shape, devices, policies,
   memory characteristics, and peer topology.

Direct paged attention is intentionally deferred because it failed quality, and
Phase 7 was excluded from the requested scope. Multi-GPU topology is detected,
but distributed tensor-parallel execution is not claimed as implemented.
