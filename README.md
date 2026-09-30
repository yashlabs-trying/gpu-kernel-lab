# GPU Kernel Lab

An end-to-end GPU-kernel and inference-engine study for
[`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B). The repository moves
from individual Triton/CUDA kernels to real prefill and decode integration,
profiling, numerical acceptance gates, and a production-style serving control
plane.

The engineering rule is simple:

> Keep a custom kernel only when it passes correctness checks and improves the
> complete workload. A faster microbenchmark alone is not a model speedup.

No external optimized kernel repository was copied into this project. PyTorch,
Triton, Transformers, cuBLAS/CUTLASS dispatch, and PyTorch SDPA/FlashAttention are
dependencies and reference implementations.

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

## Final status

The learning kernels, model integrations, static decode runtime, CUDA Graph
experiment, profiling tools, paged-KV allocator, continuous scheduler, sampling,
metrics, and OpenAI-compatible API are implemented.

Two outcomes are intentionally kept separate:

- **Accepted:** an exact-shape prefill dispatcher that enables only replacements
  that pass accuracy and end-to-end latency gates.
- **Experimental:** a highly optimized static CUDA-Graph decode path. It is much
  faster, but remains opt-in because full-path argmax agreement is 96.48%, below
  the project's 99% release requirement.

The serving control plane now has a correctness-first GPU Qwen executor. It
preserves tiled SDPA prefill, writes K/V into fixed physical pages, batches
variable-length decode requests, and passed an exact token/logit gate. A faster
direct Triton paged-attention candidate remains opt-in because it misses the 99%
decode-argmax release threshold.

## Four-model portfolio

This repository now hosts multiple optimized model projects under
[`models/`](models/):

| Slot | Model | Optimization focus | Status |
|---:|---|---|---|
| 1 | [Qwen3-0.6B](models/qwen3-0.6b/README.md) | Custom Triton/CUDA kernels, static decode, CUDA Graphs, serving control plane | Exact production path; faster experimental paths gated |
| 2 | [Llama-3.2-3B](models/optimized-llama/README.md) | vLLM serving, matched benchmarks, profiling, kernel work | Phase 1 methodology implemented; fresh GPU validation pending |
| 3 | To be selected | Architecture-specific optimization | Planned |
| 4 | To be selected | Architecture-specific optimization | Planned |

The Llama source is pinned at upstream commit `80087ed` and included directly,
so a normal clone receives it without submodule commands. See its
[provenance record](models/optimized-llama/UPSTREAM.md) and the
[portfolio roadmap](models/README.md).

## Measured final results

RTX 3090 24 GiB, PyTorch 2.8.0+cu128, Triton 3.4.0, Transformers 5.16.1,
BF16, batch 1, SDPA prefill, greedy one-token decode, context 2,048:

| Metric | PyTorch/dynamic baseline | Best implementation | Change |
|---|---:|---:|---:|
| TTFT p50 | 66.007 ms | **65.332 ms** | **1.02% lower** |
| TTFT p99* | 70.577 ms | **68.160 ms** | **3.43% lower** |
| Decode ITL p50 | 22.073 ms | **2.253 ms** | **89.79% lower** |
| Decode ITL p99* | 22.851 ms | **2.258 ms** | **90.12% lower** |
| Serial throughput from p50 ITL | 45.30 token/s | **443.90 token/s** | **9.80x** |
| Decode kernels/token | 693 | **258** | **62.8% fewer** |

`*` p99 is interpolated from five GPU-event repetitions and is preliminary, not
a statistically strong production tail-latency result.

The decode result is the **experimental CUDA-Graph path**. Its throughput is
`1000 / ITL` for a serial GPU-only loop and excludes tokenization, networking,
scheduling, EOS handling, and general sampling. It is not a production-server
throughput claim.

The real paged-serving integration was also validated on an RTX 2000 Ada (SM89):
256/256 prefill decisions and 1,024/1,024 decode decisions matched with zero
logit difference. The direct paged-attention kernel was 1.74–5.37x faster than
gather-plus-SDPA in isolation across contexts 128–4,096, but achieved only
97.65625% decode agreement and is therefore not the production default. See the
[paged-serving report](results/qwen_paged_ada_20260929/REPORT.md).

Phase 3/4 was independently checked on an RTX A4000 (SM86). The accepted
lossless paged-gather policy retained 100% of 256 prefill and 1,024 decode
argmax decisions with zero logit difference, while reducing matched paged
decode median latency by 12.84–15.46% across contexts 128–4,096. More aggressive
projection fusion did not deliver a consistent 20% without either numerical or
long-context regressions, so it remains experimental. See the
[Ampere validation report](results/qwen_paged_a4000_20260929/REPORT.md).

Phase 5/6 then hardened the production scheduler and portability boundary.
Decode-first execution, stable metadata shapes, removal of host synchronizations,
and adaptive prefill quanta raised matched concurrent throughput by 13.53% while
reducing ITL p50 by 20.85% and ITL p99 by 25.80%. A 64-request real-model run
completed without failures or leaked KV blocks at 65.52 generated token/s. See
the [portable serving report](results/qwen_phase56_a4000/REPORT.md).

| Context | Dynamic ITL | Graph ITL | Dynamic token/s | Graph token/s |
|---:|---:|---:|---:|---:|
| 128 | 21.906 ms | 2.318 ms | 45.65 | 431.46 |
| 512 | 21.958 ms | 2.321 ms | 45.54 | 430.92 |
| 2,048 | 22.073 ms | **2.253 ms** | 45.30 | **443.90** |
| 4,096 | 22.021 ms | 2.599 ms | 45.41 | 384.70 |

See the [final GPU report](results/final_newpod_20260904/REPORT.md) for methodology,
Nsight evidence, numerical results, and limitations.

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

- **476** repository tests passed in the final environment.
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
models/       self-contained optimized LLM projects and four-model roadmap
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
