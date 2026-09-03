# INT4 decode checkpoint — Qwen3-0.6B, RTX 3090

Date: 2026-09-03.

## Decision

Do not enable the all-linear INT4 adapter by default. The LM-head kernel wins
in isolation, but the whole-model decode experiment is slower and changes
predictions. Smaller packed weights do not automatically mean faster inference.
The previous BF16/RMSNorm path remains separate and unchanged.

## Restored environment

- RTX 3090, 24 GiB physical VRAM, compute capability 8.6, 82 SMs.
- Driver 580.159.03; Python 3.12.3; PyTorch 2.8.0+cu128;
  Triton 3.4.0; Transformers 5.16.1; Nsight Systems 2025.3.2.
- Model: Qwen/Qwen3-0.6B, snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Repository and virtual environment: `/workspace/gpu-kernel-lab`.
- Model and compilation caches: `/workspace/.cache`.
- Reconnect with `source /workspace/gpu-kernel-lab/scripts/activate_runpod.sh`.

The virtual environment reuses the image's system PyTorch. A replacement image
may need setup again. `/workspace` survives only according to the attached
volume's lifecycle; a directory name alone does not guarantee survival after
Pod termination. GitHub stores source and curated results, not model weights.

## What was tested

The custom W4A16 implementation packs two signed INT4 weights per byte, with
FP32 scales for groups of 128. It dequantizes inside a Triton GEMV and computes
in floating point; this is not native INT4 Tensor Core multiplication.

The model adapter replaces 197 linear modules only for single-token forwards.
Prefill remains BF16. Original weights remain resident for prefill and A/B
testing. Compact LM-head top-k is tested separately, not integrated into
Transformers generation; the model experiment still produces full logits.

Kernel source was `2a9e8ff`; the teacher-forced quality probe was added in
`c1c1abf`. Subsequent trace-summary changes do not alter kernel execution.

## Matched whole-model latency

Batch 1, BF16 activations, SDPA, cache enabled, last-token logits. Decode includes
argmax and mask growth. Two alternating runs per variant, each with two warmups
and five samples; each decode sample averages 16 token steps. Values below are
the median of all ten samples, not the mean of run medians. Quantization and
validation occur outside timing. These are local eager-workload measurements,
not serving-level TTFT or concurrent throughput measurements.

| Context | BF16 prefill ms | INT4-adapter prefill ms | BF16 decode ms/token | INT4 decode ms/token |
|---|---:|---:|---:|---:|
| 128 | 27.2701 | 28.0376 | 24.6151 | 35.9107 |
| 512 | 27.9639 | 29.2567 | 26.1302 | 35.6227 |
| 2048 | 65.6282 | 65.5386 | 25.5979 | 35.2961 |
| 4096 | 136.7956 | 137.3030 | 26.4081 | 34.5611 |

At context 2048, INT4 increases decode latency by **37.9%** and reduces serial
decode throughput by **27.5%** (39.07 to 28.33 tokens/s). Prefill is intentionally
not quantized; its small timing differences are not an INT4 prefill speedup.

## Isolated kernels on actual decode activations

Actual layer-0 and LM-head inputs captured from one model decode token. Each
microbenchmark uses 100 ms warmup and 300 ms measurement. Quantization is excluded.
These single-activation microbenchmarks are not a workload-wide speed guarantee.

| Operation | PyTorch BF16 microseconds | INT4 microseconds | BF16 / INT4 |
|---|---:|---:|---:|
| Layer 0 Q projection | 11.22 | 10.57 | 1.06x |
| Layer 0 gate projection | 16.32 | 11.36 | 1.44x |
| Layer 0 down projection | 14.57 | 21.04 | 0.69x — slower |
| LM head, full logits | 361.88 | 147.35 | 2.46x |
| LM head + top-1 | 429.65 | 160.18 | 2.68x |
| LM head + top-4 | 435.32 | 262.88 | 1.66x |
| LM head + top-8 | 435.25 | 220.48 | 1.97x |

The compact head fuses projection with tile-local selection, then uses separate
global-merge operations. It is not one launch and does not support arbitrary
logits processors, top-p, or normalized probabilities. Top-k values are checked
against the full INT4 projection, not claimed equal to BF16 predictions.

## Correctness versus quantization quality

- All **27 INT4 unit tests passed**, covering packing, dequantization, GPU GEMV,
  candidate selection, dtypes, and edge shapes.
- GPU arithmetic is checked against the dequantized-weight reference. That
  verifies implementation correctness, not preservation of original model quality.
- On the captured activation, relative L2 difference from BF16 was 6.73% for Q,
  6.77% for gate, 12.07% for down, and 13.55% for the LM head.
- Teacher-forced decode on eight original short passages, **247 tokens**:
  mean negative log likelihood increased from **3.36285 to 3.74687**;
  next-token argmax agreement was **74.90%**. Exponentiated mean NLL increased
  from 28.87 to 42.39, but this tiny custom sample is not a standard perplexity
  benchmark or an agreed acceptance threshold.
- Four chat prompts, at most 32 greedy tokens: **1/4 exact token sequences
  matched**. Different wording is not automatically wrong; the teacher-forced
  loss provides stronger evidence of degradation than exact text matching alone.
- Synthetic prefill logits matched because prefill still uses dense weights.
  The synthetic repeated-token generation check also matched; neither establishes
  quantized decode quality on realistic inputs.

## Memory accounting

Packed storage plus scales uses 0.53125 bytes/weight versus 2 for BF16:
**73.44% less per quantized weight**, or 3.76x compression, excluding metadata.
The LM-head packed buffers use 82,653,184 bytes versus 311,164,928 BF16 bytes.

However, this adapter **adds 316,616,704 bytes** of packed weights while retaining
all dense weights. It therefore increases resident GPU memory. Qwen ties the
embedding and dense LM-head weight; packing the head does not remove the dense
embedding. Do not claim whole-model memory savings from this experiment.

## Next decision gate

### Existing RMSNorm optimization still works

One additional five-sample run of the existing `qwen_rms` variant on the restored
Pod, compared with the pooled ten-sample BF16 baseline above:

| Context | RMSNorm prefill ms | Prefill latency reduction | RMSNorm decode ms/token | Decode latency reduction |
|---|---:|---:|---:|---:|
| 2048 | 53.0586 | 19.2% | 21.2038 | 17.2% |
| 4096 | 113.5688 | 17.0% | 22.1612 | 16.1% |

At 2048 this gives approximately 47.16 serial decode tokens/s, up 20.7% from
39.07. This is the existing RMSNorm benefit, not an additional INT4 gain.
The INT4 experiment did not include RMSNorm fusion. Absolute decode timings
differ from the September 1 Pod, so comparisons use the fresh baseline rather
than subtracting numbers from different hosts/runs. This RMSNorm rerun is less
replicated than the baseline/INT4 pair and is not a confidence interval.

### Nsight decode timeline

Four decode steps at context 2048, captured after warmup. Profiling overhead
changes latency; use unprofiled measurements above for speed claims.

| Trace metric, four tokens | BF16 | All-linear INT4 |
|---|---:|---:|
| Kernel launches | 6,496 | 6,496 |
| Launches per token | 1,624 | 1,624 |
| Sum of GPU kernel durations, ms | 23.3642 | 22.8487 |
| First-to-last GPU kernel span, ms | 154.0676 | 189.4979 |
| GPU kernel busy fraction of that span | 15.16% | 12.06% |

The all-linear adapter does not reduce launch fragmentation. Aggregate kernel
time improves only about 2.2%, while the traced span grows substantially. This
is consistent with increased host/dispatch overhead and inter-launch gaps, not
a large GPU-compute win. The trace alone does not attribute those gaps precisely
among Python validation, Triton dispatch, driver work, and host scheduling.

The INT4 `_gemv` launches number 788 (197/token), total 8.0922 ms over four
tokens, with launch metadata reporting 39 registers/thread, 128 threads/block,
64 shared-memory bytes, and zero local bytes/thread. These are compiler/launch
metadata, **not measured achieved occupancy, spill traffic, or bandwidth**.
Hardware-counter roofline/occupancy profiling was not performed in this run.
Reduced weight traffic does not fix the untouched norm, elementwise, attention,
mask, and KV-concatenation launches.

### Recommended sequence

1. Keep the current all-linear INT4 implementation experimental, not the default.
2. Isolate LM-head-only quantization before spending time on broader integration.
3. Improve quantization quality with measured group-size/calibration/selective
   precision experiments, including a larger fixed evaluation set.
4. Address the slow down-projection and per-layer dispatch costs before replacing
   all PyTorch linears. Measure both kernel time and end-to-end token latency.
5. Integrate compact token selection only with explicitly compatible sampling
   semantics, and benchmark against the existing RMSNorm-optimized baseline.

No 40–50% whole-model improvement is demonstrated by this INT4 checkpoint.
