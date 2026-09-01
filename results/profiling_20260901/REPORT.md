# Qwen3-0.6B optimized-kernel comparison — RTX 3090

Date: 2026-09-01. GPU medians are from five CUDA-event samples after two
warmups. Batch is 1, dtype BF16, cache enabled, SDPA attention, and only the
last-token logits are requested. Decode values are milliseconds per token over
16 steps. Candidate validation runs before timing and is outside the samples.

## Executive result

The successful optimization is Qwen-ordered RMSNorm. It lowers clean prefill
latency by 9.5--19.3% and decode latency by 13.3--14.5%. At context 2048 it
reduces prefill from 63.58 to 51.28 ms and decode from 22.11 to 19.17 ms/token.

Gate+up+SwiGLU fusion wins its isolated benchmark, especially at M=16--512, but
does not consistently improve the whole model. It improves 2048 prefill by 4.4%,
but makes decode 2.9% slower. Combining it with RMSNorm produces the fastest
prefill result, 50.54 ms at context 2048, while decode remains slower than
RMSNorm alone.

Decode GEMV is rejected: every candidate is slower than PyTorch/cuBLAS. The
Q/K-norm+RoPE+static-cache candidate is promising in isolation (4.73 us versus
120.61 us for an allocation-inclusive eager sequence), but it is not integrated
into the Transformers cache/attention path and therefore contributes zero
end-to-end speedup today.

## Environment and comparison boundary

| Item | Current run | Previous run |
|---|---|---|
| GPU | RTX 3090, 24 GiB, CC 8.6, 82 SMs | same GPU model/specification |
| Driver | 580.159.03 | 580.159.03 |
| PyTorch | 2.8.0+cu128 | 2.11.0+cu128 |
| Triton | 3.4.0 | 3.6.0 |
| Transformers | 5.16.1 | 5.16.1 |
| Model snapshot | c1899de... | same |

The current baseline is the valid denominator for candidate speedups. Previous
results are useful context but are not an A/B test because PyTorch/Triton and the
provider machine changed. Runs were sequential, not randomized/interleaved, so
small differences (roughly 1--4%) require repetition before acceptance.

## Clean model results

Lower is better. Percentages are latency reductions versus the current baseline.

### Prefill

| Context | Baseline | Qwen RMSNorm | Fused MLP | Combined |
|---:|---:|---:|---:|---:|
| 128 | 23.15 ms | 19.93 ms (13.9%) | 24.15 ms (-4.3%) | 20.60 ms (11.0%) |
| 512 | 22.27 ms | 20.16 ms (9.5%) | 24.33 ms (-9.3%) | 20.81 ms (6.6%) |
| 2048 | 63.58 ms | 51.28 ms (19.3%) | 60.79 ms (4.4%) | **50.54 ms (20.5%)** |
| 4096 | 133.67 ms | 110.80 ms (17.1%) | 131.72 ms (1.5%) | **109.10 ms (18.4%)** |

### Decode

| Context | Baseline | Qwen RMSNorm | Fused MLP | Combined |
|---:|---:|---:|---:|---:|
| 128 | 21.96 ms | **18.83 ms (14.2%)** | 22.86 ms (-4.1%) | 19.38 ms (11.8%) |
| 512 | 21.99 ms | **18.80 ms (14.5%)** | 22.77 ms (-3.5%) | 19.42 ms (11.7%) |
| 2048 | 22.11 ms | **19.17 ms (13.3%)** | 22.75 ms (-2.9%) | 19.36 ms (12.5%) |
| 4096 | 22.24 ms | **19.19 ms (13.7%)** | 22.64 ms (-1.8%) | 19.41 ms (12.7%) |

Do not add isolated speedups. The combined candidate is directly measured and
shows that fused MLP partially gives back RMSNorm's decode gain.

## Launch and traced GPU-time evidence at context 2048

Nsight covers one prefill or four decoded tokens. GPU-time sums include kernels,
not CPU gaps. Instrumented time is not the clean latency above.

| Variant | Prefill launches | Prefill GPU time | Decode launches/token | Decode GPU time/token |
|---|---:|---:|---:|---:|
| Baseline | 1593 | 62.06 ms | 1624 | 5.80 ms |
| Qwen RMSNorm | 802 | 50.00 ms | 833 | 4.41 ms |
| Fused MLP | 1509 | 58.95 ms | 1540 | 5.52 ms |
| Combined | **718** | **46.68 ms** | **749** | **4.14 ms** |

RMSNorm removes exactly `113 * 7 = 791` launches per forward by replacing eight
eager kernels with one. Fused MLP removes `28 * 3 = 84` launches by replacing
two projections plus SiLU/multiply with one projection/epilogue kernel. Combined
removes 875 launches per forward. Nevertheless, decode shows that fewer launches
do not guarantee lower clean latency when the new compute kernel is slower.

## RMSNorm: accepted as the leading candidate

Hot standalone ratios versus the eager operation sequence:

| Shape | Standalone RMSNorm | Residual + RMSNorm |
|---|---:|---:|
| 1 x 128 | 4.65x | 4.82x |
| 2048 x 128 | 3.79x | 4.04x |
| 1 x 1024 | 4.30x | 4.39x |
| 2048 x 1024 | 7.69x | 5.22x |

Residual output matched exactly in the tested cases. RMSNorm passed the complete
GPU suite and preserves Qwen's cast-before-weight boundary, but its reduction/
rsqrt implementation is not bitwise identical to PyTorch.

Model last-logit maximum absolute errors for RMSNorm were 0.40625, 0.125,
0.09375, and 0.15625 at lengths 128, 512, 2048, and 4096. Argmax and the weak
16-token synthetic generation matched at all lengths. Four natural-language
smoke prompts matched exactly for RMSNorm alone. This is encouraging, not a
production-quality equivalence evaluation.

## Gate + up + SwiGLU: useful prefill kernel, rejected for decode

The fused candidate shares X loads, retains gate/up/activation intermediates
on-chip, preserves low-precision storage boundaries, and performs one final
global write. Hot results:

| M | Eager | Fused | Latency reduction |
|---:|---:|---:|---:|
| 1 | 34.51 us | 31.59 us | 8.5% |
| 16 | 33.39 us | 23.89 us | 28.5% |
| 128 | 52.81 us | 35.49 us | 32.8% |
| 512 | 172.87 us | 106.08 us | 38.6% |
| 2048 | 520.60 us | 407.80 us | 21.7% |
| 4096 | 899.29 us | 836.57 us | 7.0% |

The candidate had zero last-logit difference in the tested model shapes and all
four natural-language prompts matched when installed alone. But clean decode
became 1.8--4.1% slower. Possible causes include weaker M=1 projection efficiency,
resource pressure, and the small absolute launch saving relative to the whole
model. Keep it for prefill experimentation; do not deploy it in decode.

## Decode GEMV: rejected

Best candidate is Split-K=4; all times are hot microseconds:

| Projection (K -> N) | PyTorch/cuBLAS | Best Triton | Triton slowdown |
|---|---:|---:|---:|
| Q, 1024 -> 2048 | 10.12 | 15.89 | 1.57x |
| K/V, 1024 -> 1024 | 7.83 | 13.10 | 1.67x |
| Gate/up, 1024 -> 3072 | 16.07 | 18.39 | 1.14x |
| Down, 3072 -> 1024 | 17.77 | 27.10 | 1.53x |

Split-K improves our one-program-K path but cannot pay back its workspace and
second reduction launch. PyTorch/cuBLAS remains the decode projection backend.
The correctness oracle uses FP32 accumulation rounded to BF16; default cuBLAS
timing retains its normal reduced-precision policy.

## Shape-specific prefill GEMM: selective only

No one kernel dominates. Representative wins at M=2048: Q projection is about
1.16x faster, K/V about 1.10x, down about 1.06x, while gate/up is approximately
tied. At M=512 gate/up is about 1.51x faster, but other shapes vary. At M=128 the
Q projection is slightly slower and down projection is substantially slower.

This supports shape-specific dispatch, not wholesale cuBLAS replacement. These
GEMMs are not installed into the model-level runs in this report.

## Q/K norm + RoPE + static cache: promising but incomplete

The isolated fused candidate measured 4.73 us versus 120.61 us for a hot,
allocation-inclusive eager Python/PyTorch sequence (about 25.5x). It removes
normalized/rotated K intermediates and writes K/V directly to a preallocated
`[B,H,position,D]` cache slot. A device-to-host bounds check was removed from the
hot path because it serialized every token; bounds validation remains optional
at API/test boundaries.

This is NOT an end-to-end speedup yet. The candidate still needs adaptation to
Transformers' exact cache layout/API, validation of attention reads after many
positions, batch/beam/paged-cache handling, and an integrated decode benchmark.
The benchmark's eager denominator includes allocations and Python indexing, so
25.5x must not be projected onto total model latency.

## Comparison with the previous profile

At context 2048, the previous PyTorch 2.11 baseline was 64.91 ms prefill and
24.59 ms/token decode. The new PyTorch 2.8 baseline is 63.58 and 22.11. The
baseline itself therefore changed by -2.1% prefill and -10.1% decode.

The previous legacy combined substitution measured 52.04 ms prefill and
20.35 ms/token decode. The new model-ordered combined candidate measures 50.54
and 19.36, nominally 2.9% and 4.9% lower. Because the framework changed, these
differences cannot be attributed solely to our new kernel code. Against the
current matched baseline, the defensible combined gains are 20.5% prefill and
12.5% decode at context 2048.

## What is still not working

1. Decode GEMV loses to cuBLAS at every model shape.
2. Fused MLP hurts end-to-end decode despite reducing 84 launches/token.
3. Combined natural-language smoke test matches only 3/4 prompts exactly. RMSNorm
   alone and fused MLP alone each match 4/4, showing small RMS perturbations can
   cross a generation decision when combined with another execution path.
4. Static KV-cache fusion is not integrated; no total decode improvement can be
   credited to it.
5. Residual+RMSNorm is tested only as a kernel; the model ablation currently
   replaces standalone norm modules and does not yet cross decoder-layer module
   boundaries to use the residual fusion.
6. Nsight Compute again returns `ERR_NVGPUCTRPERM`. Achieved occupancy, hardware
   stall reasons, DRAM/L2 throughput, and counter-derived rooflines remain
   unavailable. Nsight Systems launch counts and GPU kernel times are valid.
7. The quality probe has only four prompts. Larger deterministic evaluation and
   per-layer drift localization are required before accepting RMSNorm.

## Next action

Keep Qwen RMSNorm as the primary candidate. Integrate and measure residual-add +
RMSNorm at real decoder boundaries. Integrate the Q/K norm + RoPE + static-cache
path into a process-local cache adapter and verify attention outputs position by
position. Keep cuBLAS GEMV and SDPA. Use fused gate/up/SwiGLU only for prefill via
shape/phase dispatch until a better M=1 implementation exists.

All JSON, Nsight reports, and kernel summaries are stored beside this report.
