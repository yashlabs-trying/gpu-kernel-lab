# Final Qwen3-0.6B profiling report

Date: 2026-09-04  
GPU: NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6  
Software: PyTorch 2.8.0+cu128, Triton 3.4.0, Transformers 5.16.1  
Precision/workload: BF16, batch 1, SDPA prefill, greedy single-token decode

## Acceptance decision

Two result sets are intentionally kept separate:

- **Accepted prefill change:** exact-shape dispatch for only the 2,048-token Q
  projection. It passed kernel accuracy tests and improved integrated TTFT by
  1.02%. All other isolated GEMM winners were rejected after they failed to
  improve model-level TTFT.
- **Experimental decode path:** cross-layer RMSNorm fusion, W8 dual MLP GEMV,
  static split-KV attention, and CUDA Graph replay. It is extremely fast, but it
  failed the predefined 99% numerical agreement gate and therefore is not the
  production/default replacement.

This distinction implements the rule: a candidate is enabled by default only if
both correctness and end-to-end performance pass.

## TTFT: prefill

GPU-event time for one cache-producing forward and last-token logits:

| Context | PyTorch baseline | Whitelisted dispatcher | Difference |
|---:|---:|---:|---:|
| 128 | 28.273 ms | 29.987 ms | no active replacement; process variance |
| 512 | 29.592 ms | 32.850 ms | no active replacement; process variance |
| 2,048 | 66.007 ms | **65.332 ms** | **1.02% lower TTFT** |
| 4,096 | 136.690 ms | 136.943 ms | no active replacement; process variance |

Only `(q_proj, M=2048, K=1024, N=2048)` remains active. K/V at 4,096,
gate/up at 512, and down at 4,096 won isolated tests but regressed the integrated
model, so they are explicitly recorded in `REJECTED_AFTER_INTEGRATION`.

## ITL and serial throughput: experimental decode

Matched measurements from the CUDA-Graph benchmark process:

| Context | Dynamic ITL | Graph ITL | ITL reduction | Dynamic tok/s | Graph tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 21.906 ms | 2.318 ms | 89.42% | 45.65 | 431.46 |
| 512 | 21.958 ms | 2.321 ms | 89.43% | 45.54 | 430.92 |
| 2,048 | 22.073 ms | **2.253 ms** | **89.79%** | 45.30 | **443.90** |
| 4,096 | 22.021 ms | 2.599 ms | 88.20% | 45.41 | 384.70 |

The optimized eager path measured 16.41, 16.51, 16.41, and 16.62 ms/token for
the same contexts. CUDA Graphs remove the remaining Python, dispatcher, argument
preparation, and CPU launch gaps; they do not make each GPU kernel intrinsically
faster.

These throughput values are `1000 / ITL` for a serial GPU-only loop. They exclude
tokenization, detokenization, networking, scheduling, EOS logic, sampling, and
server overhead, and must not be presented as production serving throughput.

## Nsight Systems

At context 2,048 over 16 tokens:

| Path | Total kernels | Kernels/token | Kernel time/token | Timeline utilization |
|---|---:|---:|---:|---:|
| Dynamic hybrid | 11,088 | 693 | 4.064 ms | 14.41%* |
| Cross-layer fused CUDA Graph | 4,128 | **258** | 2.298 ms | **99.42%** |

`*` Dynamic profiling incurred severe tracing/launch overhead, so its 4.51-second
timeline span is not a latency result. Kernel counts and category structure remain
useful. Unprofiled GPU-event measurements above are the latency claims.

Cross-layer fusion reduced the earlier graph from 285 to 258 kernels/token. It
fuses 27 additional boundaries:

```text
layer i MLP output + layer i residual
        -> layer i+1 input RMSNorm
```

The first layer keeps its standalone input norm, and the last layer keeps the
terminal residual required before the model's final norm.

## Nsight Compute

Nsight Compute was invoked against `qk_norm_rope_cache_kernel`, but the RunPod
host returned:

```text
ERR_NVGPUCTRPERM: no permission to access NVIDIA GPU performance counters
```

Therefore this report does **not** invent achieved occupancy, DRAM bandwidth,
warp-stall, or roofline counters. The failure log is stored as `ncu_qk.log`.
Enabling those counters requires a host-level NVIDIA driver configuration; root
inside this container is insufficient.

## Numerical evaluation

### Kernel and integration tests

- **246 tests passed** across RMSNorm, shape-specific GEMM, split-KV attention,
  static cache, and graph-runtime suites.
- A 16-token graph/eager replay check matched every token and final position and
  retained fixed token, mask, and position addresses.
- 60 adversarial RMSNorm cases covered batch sizes 1/2/4, contexts
  1/17/128/512, and input scales from `1e-4` to `1e4`; worst output difference
  was 0.03125.

### Per-layer and prompt gate

Across 256 deterministic prompts, the RMS/fusion-only path produced:

- Maximum final-logit difference: 0.390625
- Mean final-logit difference: 0.025185
- Minimum final-logit cosine: 0.999105
- Argmax agreement: **98.05%**
- Worst logical layer cosine: 0.999726 at layer 27

The complete experimental path produced 96.48% argmax agreement. The separate
247-token teacher-forced probe produced 98.38% agreement and mean NLL 3.35573
versus 3.36071 for its dynamic hybrid reference.

The predefined requirement was at least 99% argmax agreement and at least 0.999
minimum per-layer cosine. The layer-cosine requirement passed; the argmax gate
failed. Consequently cross-layer RMSNorm and the quantized decode path remain
opt-in experiments.

The 256 prompts are deterministic synthetic probes and the 247-token set contains
eight held-out passages. Neither is a standardized downstream task evaluation.
Production acceptance still requires a real corpus/task suite and multiple GPUs.

## Implemented architecture

- Fixed `[layer,batch,kv_head,position,dimension]` K/V storage.
- Q/K RMSNorm + RoPE + direct cache/mask write.
- GQA-aware split-KV online-softmax attention with preallocated partial/LSE
  workspaces.
- Attention residual + post-attention RMSNorm fusion for all 28 layers.
- MLP residual + next-layer input RMSNorm fusion across 27 layer boundaries.
- Prefill tiled BF16 MLP versus decode W8A16 dual-GEMV MLP dispatch.
- Exact-shape prefill GEMM whitelist with PyTorch fallback.
- Fixed-address greedy decode CUDA Graph including argmax/token copy and position
  advancement.

## Final status

The performance objective was reached experimentally: 258 kernels/token and
2.253 ms ITL at context 2,048. The production correctness objective was not:
98.05% RMS-path agreement is below the 99% gate. The correct engineering decision
is to retain the code and profiler evidence while leaving the fast decode path
opt-in until numerical ordering is improved or a task-level evaluation justifies
an explicitly relaxed policy.
