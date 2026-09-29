# Qwen3-0.6B — Custom GPU kernels and production-oriented inference runtime

This project takes [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
from native PyTorch inference through kernel research, end-to-end profiling,
numerical repair, static decode, paged KV caching, continuous batching, and an
OpenAI-compatible serving control plane.

The work is built directly with PyTorch, Triton, CUDA, Transformers, cuBLAS, and
SDPA. No external optimized inference repository was copied into the Qwen path.
Every replacement follows one rule:

> A custom kernel is accepted only when it passes the numerical gate and makes
> the complete workload faster. Microbenchmark wins alone are not promoted.

## Executive summary

The repository contains two deliberately separate Qwen execution paths:

1. **Accepted serving path** — numerically exact paged-KV execution, calibrated
   per GPU architecture, continuous batching, chunked prefill, real sampling,
   capacity control, metrics, and an OpenAI-compatible API.
2. **Experimental maximum-speed path** — static cache, fused CUDA/Triton decode,
   and CUDA Graph replay. It demonstrates much larger speedups but remains
   opt-in because its historical full-path quality result is below the 99%
   release threshold.

That separation is intentional. It prevents an impressive benchmark from being
misrepresented as a production-quality replacement.

### Accepted-path highlights

| Property | Result |
|---|---:|
| SM86 prefill agreement | **100%** — 256/256 |
| SM86 decode agreement | **100%** — 1,024/1,024 |
| Accepted-path logit difference | **0.0 max / 0.0 mean** |
| SM89 independent numerical gate | **100%** — 1,000 prefill and 4,000 decode decisions |
| Paged decode latency reduction on SM86 | **12.84–15.46%** across contexts 128–4,096 |
| Concurrent-serving throughput improvement | **+13.53%** |
| Concurrent ITL p50 improvement | **20.85% lower** |
| Concurrent ITL p99 improvement | **25.80% lower** |
| 64-request generated throughput | **65.52 token/s** |
| 64-request reliability | **64 completed, 0 failed, 0 rejected** |
| KV reclamation after load | **544/544 blocks reclaimed** |

### Experimental static-path result

On an RTX 3090 at context 2,048, the static CUDA-Graph experiment reduced decode
ITL from 22.073 ms to 2.253 ms and reduced kernel launches from 693 to 258 per
token. That is a 9.80x serial GPU-loop throughput result, but the fast path is
not the serving default because its historical quality gate did not pass.

## Model architecture targeted

| Qwen3-0.6B property | Value |
|---|---:|
| Decoder layers | 28 |
| Hidden size | 1,024 |
| MLP intermediate size | 3,072 |
| Query heads | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| Vocabulary size | 151,936 |
| Attention layout | Grouped-query attention, 2 Q heads per KV head |
| Production dtype | BF16 |

These exact dimensions are part of the runtime signature. A changed model shape
cannot silently reuse a calibration produced for this model.

## System architecture

```text
OpenAI-compatible HTTP / streaming API
                    |
                    v
 request validation, priorities, cancellation, timeouts, sampling
                    |
                    v
 decode-first continuous batch scheduler
       |                            |
       |                            +--> bounded round-robin chunked prefill
       v
 batch and sequence graph-bucket selection
                    |
                    v
 capability + model-signature kernel policy
       |                 |                    |
       |                 |                    +--> exact fallback
       |                 +--> SM89 profile
       +--> SM86 profile
                    |
                    v
 Qwen paged executor
       |
       +--> fixed physical KV pages
       +--> fused page lookup + K/V gather + GQA expansion
       +--> variable-length attention masks
       +--> fixed metadata widths for stable shapes
       +--> safe PyTorch SDPA attention
                    |
                    v
 token sampler + streaming events + latency/KV metrics
```

## Work completed

### 1. Kernel laboratory

The project begins with individually testable kernels and builds toward fused
model execution:

| Kernel area | Techniques explored |
|---|---|
| Vector operations | Coalesced access, arbitrary-size masks, block/warp sweeps, effective bandwidth, add+SiLU fusion |
| RMSNorm | FP32 reductions, row mapping, register pressure, Qwen-exact cast ordering, residual fusion |
| RoPE | Physical `[B,H,S,D]` layout, paired rotations, Q/K processing, sin/cos reuse |
| Softmax | Stable max/sum reductions, load-once rows, FP32 math, masking, online-softmax concepts |
| SwiGLU | Standalone activation, dual gate/up projection, phase-specific prefill and decode designs |
| GEMM/GEMV | Tiled GEMM, shape buckets, Split-K, M=1 warp reductions, persistent/quantized experiments |
| Attention | Online tiled attention, GQA-aware split-KV decode attention, direct paged-cache reads |
| Quantization | W8A16 projections, W4A16/INT4 experiments, per-channel scaling, protected BF16 layers |

The educational implementations, explanations, and benchmarks live under
[`learning/`](../../learning/).

### 2. Prefill optimization

Prefill is treated as a matrix-heavy phase rather than reusing decode kernels.
The implementation:

- measured real Q, K, V, gate, up, and down-projection shapes;
- tested exact-shape Triton candidates against native PyTorch/cuBLAS;
- built a dispatcher that enables only measured winning shapes;
- tested residual/RMSNorm and gate/up/SwiGLU fusion;
- kept optimized PyTorch SDPA because the teaching attention kernel was slower;
- rejected isolated wins that increased complete TTFT.

The accepted prefill whitelist remains intentionally small. This is a stronger
engineering result than replacing every operation with custom Triton and hiding
the resulting regressions.

### 3. Decode optimization

Decode is handled as a launch- and memory-dominated M=1 workload. Work included:

- dedicated GEMV layouts rather than pretending decode is ordinary GEMM;
- Q/K RMSNorm + RoPE + direct cache-write candidates;
- combined QKV and gate/up projection experiments;
- W8A16 and INT4 weight-streaming experiments;
- residual + RMSNorm fusion across decoder boundaries;
- GQA-aware split-KV online-softmax attention;
- static token, position, mask, and KV-cache addresses;
- CUDA Graph capture/replay;
- kernel-launch reduction from 693 to 258 in the final static experiment.

Projection fusion was calibrated per layer. Fusing the individually safest QKV
layers together still failed composition testing at 98.8281% agreement. A
smaller three-layer candidate passed 99%, but regressed long-context latency and
was therefore not selected globally.

### 4. Numerically exact paged KV path

The accepted runtime owns physical tensors with layout:

```text
[layer, physical_block, kv_head, token_in_block, head_dimension]
```

The allocator provides:

- fixed-size reusable blocks;
- atomic admission and reservation;
- logical-to-physical block tables;
- immediate reclamation on completion, cancellation, timeout, or failure;
- per-request valid lengths for variable-length attention;
- fixed-width metadata for stable sequence buckets.

The fused gather kernel combines physical-page lookup, K/V gathering, and GQA
expansion without changing the arithmetic observed by SDPA. On SM86 this path
produced zero logit difference across the full acceptance run.

### 5. Production scheduling and serving

The control plane under [`serving/`](../../serving/) implements:

- decode-first continuous batching;
- round-robin chunked prefill;
- a measured 96-token prefill cap while decode is active;
- batch/sequence CUDA Graph bucket selection;
- capacity-aware admission control;
- request priorities, cancellation, deadlines, and graceful failure cleanup;
- temperature, top-k, top-p, repetition penalty, deterministic seeds, stop
  tokens, and optional top log-probabilities;
- OpenAI-compatible completion and chat-completion endpoints;
- JSON and server-sent-event streaming responses;
- health, model, request cancellation, and metrics endpoints;
- bounded TTFT/ITL percentile windows and KV-memory telemetry.

The scheduler originally planned decode first but the engine executed prefill
first. Real concurrent profiling exposed the mistake. Correcting execution order
and tuning the active-decode prefill quantum improved throughput and p50/p90/p99
together.

### 6. GPU portability

Portability is implemented as a correctness boundary, not a claim that one launch
configuration is optimal everywhere.

[`serving/kernel_policy.py`](../../serving/kernel_policy.py) verifies:

- CUDA compute capability;
- the exact Qwen model signature;
- architecture-specific accepted kernels;
- block and attention-tile constraints;
- calibrated per-layer fusion choices.

[`scripts/check_qwen_portability.py`](../../scripts/check_qwen_portability.py)
loads only the model configuration and reports every visible GPU, memory size,
SM count, warp size, compute capability, peer-access topology, selected policy,
and safe deployment recommendation.

Current checked-in policies:

| Architecture | Validation | Default behavior |
|---|---|---|
| SM86 / Ampere | Exact gate + latency profiling | Lossless fused paged gather |
| SM89 / Ada | Exact gate + independent 1,000-prompt RMS calibration | Exact accepted profile |
| Unknown GPU | Not calibrated | Conservative exact fallback |

The fallback keeps the model runnable and correct. It does not pretend to be
performance-calibrated for an unmeasured GPU.

## Numerical acceptance strategy

Correctness is measured at several levels:

1. Kernel-level `allclose` and adversarial-scale tests.
2. Per-layer RMSNorm drift localization.
3. Final-logit maximum and mean absolute difference.
4. Minimum cosine similarity.
5. Prefill and multi-step decode argmax agreement.
6. Teacher-forced negative-log-likelihood delta where applicable.
7. Cache ownership and complete reclamation.
8. Independent-seed runs so calibration and acceptance do not share prompts.

The accepted SM86 path produced:

```text
prefill decisions:  256 / 256
decode decisions:  1024 / 1024
maximum logit diff: 0.0
mean logit diff:    0.0
minimum cosine:     0.99999976
finite outputs:     yes
cache reclaimed:    yes
```

See the detailed [numerical methodology](../../docs/QWEN_NUMERICAL_ACCURACY.md).

## Performance results

### Accepted paged decode on RTX A4000

Stable-shape GPU-event measurement, 16 warm-up iterations and 100 measured
iterations per context:

| Context | Reference median | Accepted median | Reduction | Reference p99 | Accepted p99 |
|---:|---:|---:|---:|---:|---:|
| 128 | 33.519 ms | **29.086 ms** | **13.23%** | 44.463 ms | **34.021 ms** |
| 512 | 33.590 ms | **28.946 ms** | **13.83%** | 35.580 ms | **31.275 ms** |
| 2,048 | 33.534 ms | **29.230 ms** | **12.84%** | 46.701 ms | **31.379 ms** |
| 4,096 | 34.174 ms | **28.892 ms** | **15.46%** | 42.811 ms | **31.240 ms** |

### Real concurrent serving on RTX A4000

Sixteen simultaneous requests, mixed 32/64/128/256-token prompts, sixteen output
tokens each, maximum batch eight:

| Metric | Before scheduler repair | Final | Change |
|---|---:|---:|---:|
| Generated throughput | 57.30 tok/s | **65.06 tok/s** | **+13.53%** |
| TTFT p50 | 2,358.99 ms | **1,884.71 ms** | **-20.11%** |
| ITL p50 | 50.77 ms | **40.19 ms** | **-20.85%** |
| ITL p90 | 180.17 ms | **150.84 ms** | **-16.28%** |
| ITL p99 | 674.53 ms | **500.52 ms** | **-25.80%** |

A larger 64-request run completed all requests with zero failures and reclaimed
all 544 KV blocks. It sustained **65.52 generated token/s** with ITL
p50/p90/p99 of **40.57/182.35/407.29 ms**. TTFT includes queueing after all 64
requests are submitted simultaneously to one 16 GiB GPU.

### Experimental RTX 3090 static decode

| Metric | Dynamic baseline | Static CUDA Graph | Change |
|---|---:|---:|---:|
| ITL p50 | 22.073 ms | **2.253 ms** | **89.79% lower** |
| ITL p99* | 22.851 ms | **2.258 ms** | **90.12% lower** |
| Serial GPU-loop throughput | 45.30 tok/s | **443.90 tok/s** | **9.80x** |
| Kernels per token | 693 | **258** | **62.8% fewer** |

`*` The historical static-path p99 used five repetitions and is preliminary.
This path remains experimental because its complete-path quality gate failed.

## Repository map

```text
learning/       kernel implementations, tests, explanations, microbenchmarks
profiling/      model gates, ablations, Nsight workloads, serving load tests
serving/        allocator, scheduler, executor, sampler, metrics, API, policies
scripts/        RunPod setup, model download, smoke checks, portability preflight
docs/           experiments, accuracy methodology, RunPod and prefill plans
results/        raw JSON measurements and signed-off reports
```

Important entry points:

- [`serving/qwen_executor.py`](../../serving/qwen_executor.py) — real paged Qwen forward path
- [`serving/paged_kv.py`](../../serving/paged_kv.py) — physical KV allocator
- [`serving/scheduler.py`](../../serving/scheduler.py) — continuous scheduler
- [`serving/engine.py`](../../serving/engine.py) — decode-first execution and metrics
- [`serving/api.py`](../../serving/api.py) — OpenAI-compatible API
- [`serving/kernel_policy.py`](../../serving/kernel_policy.py) — capability policy
- [`profiling/qwen_paged_executor_quality.py`](../../profiling/qwen_paged_executor_quality.py) — exact gate
- [`profiling/qwen_serving_load.py`](../../profiling/qwen_serving_load.py) — real load test

## Reproduction

### Environment

```bash
git clone https://github.com/yashlabs-trying/gpu-kernel-lab.git
cd gpu-kernel-lab
bash scripts/setup_runpod.sh
source .venv/bin/activate
python scripts/check_environment.py
python scripts/download_model.py
python scripts/smoke_test_model.py
```

The Qwen model is public; Hugging Face Premium is not required. Keep model
weights and profiler traces on persistent storage rather than committing them.

### Unit and integration tests

```bash
pytest -q
```

### Portability preflight

```bash
python scripts/check_qwen_portability.py \
  --local-files-only \
  --output results/portability.json
```

### Numerical gate

```bash
python profiling/qwen_paged_executor_quality.py \
  --prompts 256 \
  --decode-steps 4 \
  --local-files-only \
  --output results/qwen_quality.json
```

### Concurrent serving benchmark

```bash
python profiling/qwen_serving_load.py \
  --requests 64 \
  --prompt-lengths 32 64 128 256 \
  --output-tokens 16 \
  --max-batch-size 8 \
  --prefill-chunk 128 \
  --prefill-budget 512 \
  --decode-prefill-chunk 96 \
  --local-files-only \
  --output results/qwen_serving_load.json
```

## Evidence and reports

- [Final kernel/model profiling](../../results/final_newpod_20260904/REPORT.md)
- [Static decode runtime](../../results/decode_runtime_20260904/REPORT.md)
- [Serving system](../../results/final_serving_20260904/REPORT.md)
- [SM89 paged-executor validation](../../results/qwen_paged_ada_20260929/REPORT.md)
- [SM86 paged-executor Phase 3/4](../../results/qwen_paged_a4000_20260929/REPORT.md)
- [Portable serving Phase 5/6](../../results/qwen_phase56_a4000/REPORT.md)

## Scope and honest limitations

- The accepted result is a production-oriented **single-GPU serving core**, not
  a claim of hyperscale operational readiness.
- The real concurrent load test excludes tokenization, HTTP transport, TLS,
  network latency, and multi-process orchestration.
- GPU topology is detected, but distributed tensor-parallel execution has not
  been implemented or validated.
- Direct paged attention remains disabled because it did not pass the numerical
  gate; the accepted path uses fused gather plus PyTorch SDPA.
- The extremely fast CUDA-Graph result is experimental and is never selected by
  the production policy.
- Latency results are hardware- and software-stack-specific. Re-run calibration
  and acceptance on every new architecture.

These limitations are enforced in code through exact fallbacks and opt-in
experimental switches rather than hidden in benchmark notes.
