# Decode stage 2 — native CUDA, selective INT8, and split-KV

Date: 2026-09-03. Model: Qwen3-0.6B, snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`. RTX 3090, BF16 activations,
PyTorch 2.8.0+cu128, Triton 3.4.0, Transformers 5.16.1.

## Decision

The selective combination is the strongest **quality-first candidate** in this
experiment: approximately 15–17% lower decode latency than RMSNorm alone, with
98.79% next-token agreement on the small held-out probe. The all-INT8 body also
improves decode and is slightly faster at context 4096, but has more measured
quality drift. Neither is production-validated. INT4 LM head remains optional:
its small isolated model-level gain does not clearly justify the extra drift.

## Repeated whole-model results

Batch 1, BF16 activations, SDPA, dynamic KV cache, last-token logits. Every sample
averages 16 decode steps and includes argmax/mask growth. Two warmups and five
samples per run. The RMSNorm control has four runs (20 samples); INT4-head,
INT8-body and selective candidates have two runs each (10 samples). INT8+INT4
has only one run (five samples) and is less replicated. Values are pooled
medians of **all samples**, not selected best runs. See `aggregate.json` for
run medians, ranges, filenames and formulas. Runs were sequential on the same
Pod, not randomized or statistically certified; host timing variation remains.

| Candidate | Decode at 2048, ms/token | Reduction vs RMSNorm | Decode at 4096, ms/token | Reduction vs RMSNorm |
|---|---:|---:|---:|---:|
| RMSNorm only, control | 23.449 | reference | 23.141 | reference |
| RMSNorm + INT4 head | 22.827 | 2.65% | 22.824 | 1.37% |
| RMSNorm + INT8 body, BF16 head | 20.491 | 12.61% | 18.950 | 18.11% |
| RMSNorm + INT8 body + INT4 head | 18.950 | 19.19% | 19.328 | 16.48% |
| Selective combination, BF16 head | 19.542 | 16.66% | 19.604 | 15.29% |

For the selective combination, serial decode throughput is approximately
**51.17 vs 42.65 tokens/s** at context 2048, an increase of **20.0%**. At 4096 it
is 51.01 vs 43.21 tokens/s, an increase of 18.1%. These are inverse token
latencies, not concurrent serving-throughput measurements. Small head-only
differences are particularly sensitive to the observed timing variability.

Prefill is effectively unchanged: RMSNorm control vs selective candidate is
52.739 vs 52.772 ms at 2048 and 113.624 vs 113.707 ms at 4096. These adapters
target decode only. This is not a new serving-TTFT improvement.

Do not add these percentages to an earlier RMSNorm percentage. This stage uses
RMSNorm **already enabled** as its reference, and the Pod's absolute decode
latency varied from the earlier same-day baseline. No 50% end-to-end gain is
claimed. The fastest isolated operation is not necessarily the best model path.

## What we built

1. Native CUDA BF16/W8A16 GEMV: one warp per output row, FP32 accumulation,
   `__shfl_down_sync` reductions, no cross-warp shared-memory reduction.
2. Scheduling experiments: 1/4/8 independent warps per block and a persistent
   grid capped at twice the SM count. No single configuration wins every shape.
3. Activation-aware INT8 channel scaling with local calibration-MSE selection
   over alpha 0/0.5/1. This is **AWQ-inspired**, not a complete AWQ implementation,
   AWQ checkpoint format, W8A8, or native INT8 Tensor Core computation.
4. Combined QKV projection, plus gate/up projection with a fused BF16-boundary
   SiLU-and-multiply epilogue. No bias fusion is needed for this bias-free model.
5. RMSNorm + isolated INT4 LM-head model ablation, leaving all other linears BF16.
6. A separate GQA split-KV attention prototype, stable log-sum-exp merge, and
   masked tail chunks. It is **not installed into the model**.
7. A selective model candidate: BF16 combined QKV, INT8 gate/up/SwiGLU, original
   PyTorch output/down projections, BF16 LM head, and existing RMSNorm fusion.

All candidates remain opt-in. The default path is unchanged. See
`learning/10_cuda_decode/README.md` for the mapping, formulas, restrictions, and
reproduction commands.

## Held-out quality: INT8 is substantially closer than the prior INT4 experiment

Same 247 teacher-forced tokens and four greedy chat prompts for all candidates.
Calibration uses four different passages. Lower mean NLL is better, but this
small sample is not a standard benchmark or a production acceptance threshold.

| Variant | Mean NLL | Next-token agreement with BF16 | Exact chat outputs |
|---|---:|---:|---:|
| Original BF16 reference | 3.36285 | reference | reference |
| Previous all-linear INT4, without RMS fusion | 3.74687 | 74.90% | 1/4 |
| RMSNorm + INT4 head only | 3.41632 | 88.26% | 3/4 |
| RMSNorm + INT8 body, BF16 head | 3.36823 | 96.36% | 4/4 |
| RMSNorm + INT8 body + INT4 head | 3.41440 | 89.88% | 3/4 |
| Selective combination, BF16 head | 3.36071 | 98.79% | 4/4 |

The INT4 head still introduces measurable drift. For a quality-first candidate,
keep the head BF16. Do not interpret the selective candidate's tiny NLL decrease
as a demonstrated quality improvement. Exact chat matches also do not replace
a larger held-out evaluation. Historical INT4 values come from the earlier
same-day report, not a new all-linear INT4 rerun.

## Isolated projection results

Actual first-layer decode activation, separate calibration prompts. Hot-buffer
microbenchmarks, 50 ms warmup and 150 ms measurement per configuration; packing,
calibration and compilation excluded. Best-of-configuration results are tuning
evidence, not an independently validated speed guarantee. The original PyTorch
operators are the reference; there is no external hand-tuned kernel repository
plugged into the candidate.

| Operation | PyTorch microseconds | Best native BF16 microseconds | Best native INT8 microseconds |
|---|---:|---:|---:|
| Q | 11.58 | 10.43 | 12.39 |
| K | 8.17 | 7.55 | 11.04 |
| V | 8.11 | 7.54 | 11.39 |
| Output projection | 10.69 | 10.95 | 18.06 |
| Gate | 15.95 | 13.90 | 14.22 |
| Up | 15.94 | 13.84 | 14.23 |
| Down | 14.00 | 13.85 | 25.10 |
| Combined QKV | 21.59 for three linears | 16.69 | 20.56 |
| Gate/up/SwiGLU | 34.03 for eager chain | 22.56 | 20.60 |

Combined BF16 QKV reduces isolated latency by 22.7%; INT8 gate/up/SwiGLU by
39.4%. INT8 output/down projections lose badly despite smaller weights. This
motivates the selective candidate instead of replacing every linear by default.
The tiny BF16 down-projection difference is not a robust win claim.

Across the first-layer probes, INT8 relative L2 drift from BF16 is approximately
0.43–0.76%, compared with much larger drift in the previous INT4 probes (which
used a different captured activation, so they are not matched per-tensor A/B).
The full INT8-body calibration selected alpha 0.5 for 109 packed matrices,
alpha 0 for two, and alpha 1 for one. That measures a local calibration choice,
not independently proven generalization.

## Split-KV attention: separate candidate

Synthetic Qwen head shapes: 16 Q heads, 8 KV heads, D=128, one query. SDPA uses
GQA and attends to all supplied valid past/current cache positions. Allocation
and the two candidate kernel launches are included in these hot microbenchmarks.

| Cache length | SDPA microseconds | Best split-KV microseconds | Latency reduction |
|---|---:|---:|---:|
| 128 | 15.22 | 10.82 | 28.9% |
| 2048 | 29.11 | 22.22 | 23.7% |
| 8192 | 65.98 | 53.25 | 19.3% |

These are not model-level improvements. The prototype has no padding/window
mask, dropout, or arbitrary query positions; every cache position supplied must
be valid. It uses per-chunk normalized outputs and log-sum-exp, followed by a
weighted merge. Summing partial outputs directly would be incorrect. The KV
cache stays in global VRAM; it is not stored wholesale in shared memory.

## Memory and implementation limits

### Correctness and Nsight evidence

All **44 tests pass**: calibration/packing, BF16 and INT8 projection arithmetic,
fused SwiGLU, independent-warp and persistent schedules, output tails,
non-default stream, GQA, cache-stride/tail handling, and stable softmax at D64/128.
Numerical checks use FP32/dequantized references, not the original unquantized
weight matrix as the kernel-correctness oracle.

Nsight captures four decode steps at context 2048:

| Metric | RMSNorm reference | Selective combination |
|---|---:|---:|
| Kernel launches per token | 833 | 693 |
| Total launches, four steps | 3332 | 2772 |
| Sum of GPU kernel durations, four steps | 18.1219 ms | 15.8724 ms |
| First-to-last kernel span under profiling | 123.6059 ms | 99.0103 ms |

The combination eliminates **140 launches/token (16.8%)**: two QKV launches and
three gate/up/activation launches per layer, across 28 layers. Summed kernel
time falls about 12.4% in the trace. Profiler spans include substantial overhead;
use the unprofiled model table for speed claims. Most span time remains outside
GPU kernel execution, so host dispatch and cache management are still important.

Launch metadata for combined BF16 QKV reports 36 registers/thread, 128
threads/block, zero shared bytes and zero local bytes/thread. INT8 fused
gate/up/SwiGLU reports 47 registers/thread, 256 threads/block, zero shared bytes
and zero local bytes/thread. These are **not achieved occupancy or measured
spill traffic**. Fused projection durations include their epilogues. Categories
in the generated summary use kernel-name heuristics, not semantic attribution.

### Storage is not resident model memory

INT8 storage is 1 byte per weight plus FP32 group/channel scales. The full INT8
body adds **454,967,296 bytes** to this A/B model because original BF16 weights
remain for prefill. The hybrid also retains original weights and adds concatenated
BF16 QKV buffers. None of these adapters demonstrates reduced resident model
memory. The INT4 head also retains the tied dense embedding/head weight.

The kernels use coalesced scalar loads, not explicit vectorized packed loads.
Weights are dequantized to floating point. Persistent scheduling and one-warp
blocks are experiments, not automatic upgrades. Group/shape specialization and
vectorized dequantization are still possible future improvements. No achieved
occupancy, spill bandwidth, or roofline counters are claimed here.

The model adapters are eager, batch-one, single-token experiments. They are not
concurrent-forward safe, not a graph-captured serving implementation, and not
tested on another architecture. Changing GPU requires extension rebuild and
new measurements. Prefill still uses the existing dense projections and RMSNorm.

## Next practical steps

- Keep BF16 LM head for the quality-first path; the INT4 head remains optional.
- Run a larger fixed held-out corpus and task suite before promoting INT8.
- Validate split-KV against real model cache layouts/masks before integration.
- Target static KV allocation and CUDA Graphs next: reducing projection launches
  does not eliminate the remaining Python dispatch and cache-management overhead.
- Optimize or avoid the losing INT8 output/down projections; do not add their
  standalone speed claims to a whole-model improvement percentage.

References: [AWQ](https://github.com/mit-han-lab/llm-awq),
[NVIDIA warp primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/),
[Flash-Decoding](https://pytorch.org/blog/flash-decoding/).
