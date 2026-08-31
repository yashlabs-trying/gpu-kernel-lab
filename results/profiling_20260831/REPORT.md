# Qwen3-0.6B RTX 3090 profiling report

Date: 2026-08-31. This report separates measured facts, analytical estimates,
and proposals. Raw JSON/CSV/Nsight reports are stored beside it.

## Executive summary

- Qwen3-0.6B loaded and generated correctly in BF16.
- Clean generation-style prefill latency was 28.09, 26.67, 64.91, and 135.84 ms
  at 128, 512, 2,048, and 4,096 tokens respectively.
- Clean decode stayed near 24.3--25.1 ms/token from 128 through 4,096 context.
- At 2,048 tokens, prefill kept the GPU timeline busy (94.9% of the interval
  between first and last kernel). Decode was launch fragmented (14.2% traced
  busy fraction), though tracing changes timing and this is not SM occupancy.
- PyTorch already selected CUTLASS/Ampere GEMMs and PyTorch FlashAttention.
- In the prefill trace, GEMM/GEMV kernels consumed 29.68 ms (47.4% of summed
  kernel time), FlashAttention 8.77 ms (14.0%), and miscellaneous elementwise
  work 14.95 ms (23.9%).
- Q/K RMSNorm alone used about 8.40 ms and 448 GPU kernels over 28 layers.
- All 150 teaching-kernel tests passed. They are still not automatically
  model-exact or model-integrated.
- Native comparisons changed the conclusion: our attention beat materialized
  eager attention but lost to PyTorch SDPA. Our default GEMM was shape-sensitive.
- Experimental RMSNorm + SwiGLU substitution reduced 2,048-token prefill from
  64.91--65.11 to 52.04 ms, and decode from 24.59--28.12 to 20.35 ms/token.
  One of four natural-language generations changed; equivalence is NOT established.
- Nsight Compute hardware counters were blocked by `ERR_NVGPUCTRPERM`; achieved
  occupancy, stalls, DRAM traffic, and counter-derived rooflines are unavailable.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6, 82 SMs |
| Driver | 580.159.03 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / CUDA 12.8 |
| Transformers / Triton | 5.16.1 / 3.6.0 |
| Nsight Systems | 2025.3.2 |
| Nsight Compute | 2025.1.1 (counters permission-blocked) |
| Model | Qwen/Qwen3-0.6B, snapshot c1899de... |
| Model dtype | BF16 |
| Attention backend | SDPA, resolved to PyTorch FlashAttention kernels |

## Measurement definitions

The clean workload uses batch 1, cache enabled, last-token logits, two warmups,
five CUDA-event samples, and deterministic token IDs. Prefill includes KV-cache
construction. Decode includes one-token model forwards, mask growth, argmax, and
cache updates; cache construction is outside decode timing.

The historical baseline (`baseline_20260831_original.json`) is a different
workload: prefill disables the cache and computes all logits. It is retained but
must not be compared as if it were the same operation.

Nsight traces capture one measured iteration after warmup. Annotated traces add
hook overhead. `kernel_busy_fraction_of_span` is the union of kernel intervals
divided by first-to-last-kernel time. It is NOT achieved SM occupancy.

Scope: this is Qwen3-0.6B, not a 7--8B model, on one GPU at batch 1. We tested
forward inference, not training/backward, serving concurrency, quantization,
vLLM, or every possible sequence length. CUTLASS/cuBLAS evidence comes from
PyTorch dispatch and native comparisons, not a separately tuned CUTLASS runner.

## Architecture and computation map

The checkpoint has 28 layers, hidden width 1,024, MLP width 3,072, 16 query
heads, 8 KV heads, head dimension 128, vocabulary 151,936, and tied embedding/
LM-head weights. Qwen uses Q/K head RMSNorm and RoPE theta 1,000,000.

For batch 1, let T be the number of input tokens: T=S in prefill and T=1 in
decode. Each layer performs:

| Operation | Shape / work | Main optimization question |
|---|---|---|
| Input RMSNorm | T x 1024 | fuse residual/norm, preserve rounding |
| Q projection | [T,1024] @ [1024,2048] | Tensor Core reuse in prefill; weight streaming in decode |
| K and V projections | two [T,1024] @ [1024,1024] | combine projections or launches |
| Q/K head norm | T x 16 x 128 and T x 8 x 128 | many small reductions; fuse with RoPE |
| RoPE | rotate Q/K pairs using position tables | correct pairing/strides/theta; reuse tables |
| Cache update | append K/V for each layer | avoid recopying growing cache |
| Causal GQA attention | 16 Q heads share 8 KV heads | tiled prefill; KV streaming during decode |
| Output projection | [T,2048] @ [2048,1024] | GEMM/GEMV selection |
| Residual + post-attention norm | T x 1024 | fused residual + norm |
| Gate and up projections | two [T,1024] @ [1024,3072] | combine GEMMs, fuse activation epilogue |
| SwiGLU | SiLU(gate) * up, T x 3072 | remove intermediate traffic |
| Down projection + residual | [T,3072] @ [3072,1024] | GEMM then residual fusion |

Final RMSNorm and last-token LM-head projection follow the 28 layers. The
BF16 KV cache alone is `28 * 2 * S * 8 * 128 * 2` bytes: 224 MiB at S=2048,
448 MiB at S=4096, excluding temporary copies and additional decoded tokens.

## Clean model latency

| Phase | Initial context | GPU median | Wall median | Peak allocated |
|---|---:|---:|---:|---:|
| Prefill | 128 | 28.09 ms | 28.12 ms | 1.14 GiB |
| Prefill | 512 | 26.67 ms | 26.70 ms | 1.19 GiB |
| Prefill | 2,048 | 64.91 ms | 64.93 ms | 1.39 GiB |
| Prefill | 4,096 | 135.84 ms | 136.16 ms | 1.66 GiB |
| Decode | 128 | 24.33 ms/token | 24.33 ms/token | 1.13 GiB |
| Decode | 512 | 25.12 ms/token | 25.13 ms/token | 1.20 GiB |
| Decode | 2,048 | 24.59 ms/token | 24.60 ms/token | 1.34 GiB |
| Decode | 4,096 | 25.07 ms/token | 25.08 ms/token | 1.57 GiB |

The 128/512 non-monotonic prefill result shows that two warmups do not fully
control clock/cache state at small sizes. We use the result as a baseline, not a
high-precision claim about a sub-millisecond difference.

## Nsight Systems: prefill

The 2,048-token trace contained 1,593 kernel launches, 62.59 ms summed/union
kernel time, a 65.97 ms first-to-last-kernel span, and a 94.9% busy fraction.
There was no kernel overlap in this trace (sum equals union).

| Coarse kernel-name category | Kernel time | Share |
|---|---:|---:|
| GEMM/GEMV | 29.68 ms | 47.4% |
| Miscellaneous elementwise | 14.95 ms | 23.9% |
| PyTorch FlashAttention | 8.77 ms | 14.0% |
| Copy/cast | 4.51 ms | 7.2% |
| Concatenation | 2.91 ms | 4.7% |
| Reduction | 1.76 ms | 2.8% |

Name classification is approximate; detailed raw names remain in CSV.

NVTX/module attribution of kernel time:

| Model operation (28 layers unless noted) | Time | Kernels |
|---|---:|---:|
| MLP gate projection | 6.28 ms | 28 |
| MLP up projection | 6.20 ms | 28 |
| Q head RMSNorm | 5.56 ms | 224 |
| MLP down projection | 5.03 ms | 28 |
| Q projection | 4.21 ms | 28 |
| Attention output projection | 3.45 ms | 28 |
| K head RMSNorm | 2.84 ms | 224 |
| Post-attention RMSNorm | 2.81 ms | 224 |
| Input RMSNorm | 2.81 ms | 224 |
| K projection | 2.07 ms | 28 |
| V projection | 2.07 ms | 28 |
| SiLU module | 0.88 ms | 28 |
| Final LM head | 0.36 ms | 1 |

Why an RMSNorm produces eight kernels: the eager implementation separates FP32
conversion, square, reduction, epsilon/add, rsqrt, multiply, dtype conversion,
and weight multiplication. Fusing it is high-confidence traffic/launch removal.

Representative library kernels include:

- CUTLASS tensor-op BF16 GEMM (`cutlass_80_tensorop...`), 224 registers/thread,
  73,728 B shared memory, 256 threads/block.
- Ampere BF16 GEMMs, 222--234 registers/thread and 32--72 KiB shared memory.
- PyTorch FlashAttention forward, 184 registers/thread, 49,152 B shared memory,
  128 threads/block.

Those are launch-resource metadata, not measured occupancy. High registers and
shared memory constrain possible resident blocks, but hardware counters are
required to state achieved occupancy and the actual limiting resource.

## Nsight Systems: decode

The unannotated 2,048-context trace covers four decoded tokens. It contained
6,496 kernel launches: 1,624 launches/token. It recorded 22.90 ms kernel time
(5.72 ms/token) across a 161.42 ms traced span, for a 14.2% busy fraction.
Tracing raised wall time substantially versus the clean 24.59 ms/token result,
so use the trace for composition/fragmentation, not absolute decode latency.

Per-token kernel time from the trace:

| Category | Per token |
|---|---:|
| GEMM/GEMV | 2.04 ms |
| FlashAttention split-KV | 0.74 ms |
| Miscellaneous elementwise | 1.44 ms |
| Concatenation | 0.74 ms |
| Copy/cast | 0.44 ms |
| Reduction | 0.32 ms |

Observed decode kernels include cuBLAS GEMV, an Ampere sliced GEMM, split-KV
FlashAttention, and many concatenate/copy/elementwise kernels. Decode is therefore
not simply “slow attention”: launch fragmentation, weight-streaming GEMVs, cache
growth/copies, and small operations all contribute.

## Roofline thinking (analytical, not counter-derived)

Arithmetic intensity is useful FLOPs divided by bytes transferred. The Roofline
upper bound is `min(peak FLOP/s, memory bandwidth * intensity)`. Actual traffic
depends on cache reuse and tiling; the estimates below assume each listed tensor
is read/written once and therefore are optimistic.

| Operation | Ideal arithmetic intensity | Likely issue |
|---|---:|---|
| FP32 vector add | 1 / 12 = 0.083 FLOP/B | DRAM bandwidth for large N; launch latency for small N |
| BF16 decode GEMV, M=1 | approximately 1 FLOP/B | weight bandwidth plus launch latency |
| Q projection prefill, M=N=2048,K=1024 | 512 FLOP/B | Tensor-compute/tile-resource limited if reuse succeeds |
| Gate/up GEMM, M=2048,N=3072,K=1024 | about 559 FLOP/B | Tensor compute and tile resources |
| Prefill attention, ideal S=2048 | about S/2 = 1024 FLOP/B | Tensor compute, softmax/reductions, tile reloads/resources |
| Decode attention, one query | approximately 1 FLOP/B | KV-cache bandwidth/latency and split reduction |

The full decode must stream approximately 0.6B BF16 weights (about 1.2 GB) unless
caches retain subsets; this dwarfs L2. Its low-M GEMV geometry offers much less
reuse than prefill GEMM. The clean 25 ms/token cannot be explained from this
ideal byte count alone because dispatch gaps and many small operations matter.

## Nsight Compute limitation

Even as root inside the Pod, Nsight Compute returned `ERR_NVGPUCTRPERM`. This is
a host/provider security setting. We did not bypass it. Consequently this report
does NOT contain measured:

- achieved occupancy or active warps;
- SM/Tensor utilization percentages;
- DRAM/L2 throughput or cache hit rate;
- eligible-warps/stall reasons;
- runtime local-memory spill traffic;
- Nsight counter-derived roofline points.

Nsight documentation describes Roofline and occupancy interpretation at
https://docs.nvidia.com/nsight-compute/ProfilingGuide/ . A future Pod with counter
permission should run the saved capture workloads with the `detailed`/`roofline`
sets rather than invent these values.

## Teaching-kernel validation

All existing GPU tests passed:

| Lesson | Tests |
|---|---:|
| Vector add | 37 |
| SwiGLU | 33 |
| RMSNorm/residual RMSNorm | 21 |
| RoPE | 9 |
| Softmax | 22 |
| GEMM | 16 |
| Tiled attention | 12 |
| Total | 150 |

These tests cover selected dtypes/shapes/tolerances. They do not cover Qwen GQA
integration, KV-cache semantics, all strides, full generation equivalence, or
training gradients.

## Stronger native-backend comparison

These hot CUDA-graph microbenchmarks replay 16 calls per sample. Data are hot and
allocation patterns are captured. They are designed to reduce Python launch
effects; they are not cold-memory or end-to-end measurements.

| Case | Native/reference | Our Triton | Ratio (native / Triton) | Result |
|---|---:|---:|---:|---|
| RMSNorm 1x1024 | PyTorch native 2.50 us | 1.79 us | 1.40x | Triton faster |
| RMSNorm 512x1024 | PyTorch native 3.71 us | 3.01 us | 1.23x | Triton faster |
| RMSNorm 2048x1024 | PyTorch native 11.68 us | 10.53 us | 1.11x | Triton faster |
| SwiGLU 512x3072 | eager 16.70 us | 11.84 us | 1.41x | fusion faster |
| SwiGLU 2048x3072 | eager 79.42 us | 47.17 us | 1.68x | fusion faster |
| Softmax 128x1000 | PyTorch native 5.50 us | 2.69 us | 2.04x | Triton faster |
| Softmax 128x3000 | PyTorch native 5.89 us | 4.22 us | 1.40x | Triton faster |
| Attention 1x4x128x128 | SDPA 9.02 us | 11.62 us | 0.78x | SDPA faster |
| Attention 1x4x512x128 | SDPA 19.65 us | 31.94 us | 0.62x | SDPA faster |
| Q GEMM M=1,N=2048,K=1024 | PyTorch 7.46 us | 14.11 us | 0.53x | PyTorch faster |
| Q GEMM M=2048,N=2048,K=1024 | PyTorch 163.20 us | 134.78 us | 1.21x | Triton faster in this hot test |

Our attention's win over materialized eager attention is not a win over the
actual optimized backend. Do not replace SDPA with it yet. GEMM is shape-specific:
the broader lesson benchmark found PyTorch slightly faster at 256/1024 square
shapes and Triton only 1.02x faster at 4096x4096x1024.

## BF16 GEMM precision finding

At M=512,N=2048,K=1024, direct comparison to default PyTorch exceeded the chosen
tolerance. An FP32-reference probe found:

- default PyTorch BF16 reduced-precision reduction: mean absolute error 0.02543,
  not within the probe's 0.03 allclose rule;
- Triton FP32 accumulator: mean absolute error 2.84e-5, within the rule;
- PyTorch with BF16 reduced-precision reduction disabled: the same small error
  level as Triton.

Thus the initial mismatch is primarily a precision-policy mismatch, not evidence
that Triton's result is worse. Performance comparisons must declare this policy.

With BF16 reduced-precision reductions disabled for the matched-policy repeat,
all tested GEMM shapes passed the chosen tolerance:

| M,N,K | PyTorch | Triton | Native / Triton |
|---|---:|---:|---:|
| 1,2048,1024 | 7.424 us | 14.016 us | 0.53x |
| 512,2048,1024 | 43.648 us | 45.568 us | 0.96x |
| 2048,2048,1024 | 163.136 us | 132.352 us | 1.23x |

## Custom-kernel resource/codegen evidence

Representative Systems launch metadata:

| Kernel | Registers/thread | Shared memory | Threads/block |
|---|---:|---:|---:|
| Vector add | 16 | 0 | 128 |
| SwiGLU | 16 | 0 | 128 |
| RoPE | 22 | 0 | 128 |
| RMSNorm | 26 | 16 B | 128 |
| Residual RMSNorm | 29 | 16 B | 128 |
| Softmax | 24 | 16 B | 128 |
| GEMM | 79 | 16 KiB | 128 |
| Tiled attention | 110 | 42 KiB | 128 |

For representative GEMM generated PTX, compiler metadata reported zero spills,
20 `cp.async` mentions, and 16 `mma.sync` mentions. This confirms async-copy and
Tensor Core instruction generation for that compiled configuration. It does not
measure their utilization or prove that another shape/configuration has the same
codegen.

## Did our kernels improve the model?

Yes, in experimental full-model runs. We replaced 113 RMSNorm modules and/or
28 MLP activation paths using process-local forward overrides. The checkpoint
and default model implementation were not changed. No new advanced kernel was
written for these tests: these are the existing teaching kernels.

GPU medians in ms; decode values are per token. Baseline ranges show the initial
and repeated runs, NOT confidence intervals. Ablations were sequential, not
randomized/interleaved, so clock/host variability remains a confounder.

| Phase | Context | Baseline range | RMS only | SwiGLU only | Both |
|---|---:|---:|---:|---:|---:|
| Prefill | 128 | 27.84--28.09 | 24.94 | 31.20 | 21.74 |
| Prefill | 512 | 26.67--28.14 | 24.53 | 30.57 | 21.63 |
| Prefill | 2048 | 64.91--65.11 | 52.89 | 64.51 | 52.04 |
| Prefill | 4096 | 134.95--135.84 | 113.22 | 134.33 | 111.17 |
| Decode | 128 | 24.33--27.69 | 22.28 | 28.59 | 20.35 |
| Decode | 512 | 25.12--28.22 | 22.24 | 28.05 | 20.90 |
| Decode | 2048 | 24.59--28.12 | 22.31 | 27.83 | 20.35 |
| Decode | 4096 | 25.07--28.10 | 22.32 | 27.94 | 19.82 |

At S=2048 both gave about 1.25x prefill speedup (20% lower latency) and
1.21--1.38x decode speedup (17--28% lower latency). RMSNorm accounts for most
of the convincing large-prefill gain. SwiGLU alone did not demonstrate a clear
end-to-end win despite its microbenchmark win. Do not add isolated speedups.

Nsight confirms launch removal at S=2048:

| Trace | Baseline launches | Both launches | Summed GPU kernel time, baseline -> both |
|---|---:|---:|---:|
| Prefill, one forward | 1593 | 774 | 62.59 -> 49.53 ms |
| Decode, four tokens | 6496 | 3220 | 22.90 -> 17.47 ms |

That is 819 fewer launches per forward: `113 * (8 - 1) + 28`. Decode drops
from 1624 to 805 launches/token. These traces support actual work removal;
their instrumented durations are not substitutes for the clean timings above.

### Accuracy gate: not ready for unconditional replacement

- Synthetic last-token argmax agreed at all four lengths, and 16 greedy tokens
  matched, but that synthetic generation repeats a token and is a weak test.
- For both substitutions, maximum absolute last-logit error was 0.6875 at
  length 128 and 0.125--0.15625 at the longer lengths. Cosine similarity was
  about 0.99948--0.99994; similarity is not equivalence.
- Four natural-language prompts, up to 32 greedy new tokens: **3/4 exact
  generation matches**. The GPU-explanation response changed wording. This
  does not establish quality degradation, but proves the outputs can differ.
- Existing RMSNorm computes weight multiplication before the final cast, while
  Qwen rounds normalized values before multiplying the weight. Fused SwiGLU
  also removes an eager low-precision rounding point. These are known semantic
  differences; accumulation/reduction order can add further differences.

Next correctness work must pin these semantics, compare per-layer activations
and logits, and evaluate a larger real-prompt/quality dataset before accepting
the speedup as a production improvement.

## Ranked next work

1. Model-exact RMSNorm fusion: Q/K RMSNorm alone has 448 launches and 8.40 ms at
   S=2048. Preserve Qwen rounding/weight semantics and fuse Q/K normalization
   with RoPE/cache writes where possible.
2. Decode dispatch: 1,624 launches/token in the trace. Use CUDA Graphs/static
   cache addresses, avoid repeated mask concatenation, and fuse residual/norm and
   activation chains. Measure clean latency after each change.
3. Keep PyTorch SDPA/FlashAttention as prefill attention baseline. Adapt our
   attention for 16Q/8KV GQA and skip causal tiles before considering replacement.
4. Shape-specific GEMM selection: keep cuBLAS/CUTLASS for shapes it wins; autotune
   only exact Qwen prefill shapes and use GEMV-specific strategies for decode.
5. Fused gate+up projection and SwiGLU; compare against CUTLASS epilogues and
   torch.compile, not only separate eager operations.
6. Obtain a counter-enabled GPU to validate DRAM throughput, achieved occupancy,
   stalls, and hierarchical roofline positions.

## Reproducibility artifacts

- `model_sdpa.json`: clean baseline samples and exact config.
- `model_baseline_repeat.json`: repeat baseline to expose run variability.
- `model_triton_{rms,swiglu,both}.json`: full-model ablations and logit checks.
- `natural_language_drift.json`: four real-prompt generation comparisons.
- `*_stats_*.csv`: Nsight summaries.
- `*.nsys-rep`: raw prefill/decode/custom-kernel traces.
- `timeline_summary.json`: interval/category/resource summary.
- `native_comparison.json`: hot native/Triton comparisons and correctness.
- `native_comparison_strict.json`: matched BF16 reduction-policy comparisons.
- `gemm_accuracy_probe.json`: BF16 precision-policy diagnosis.
- `triton_gemm_codegen.json` and `triton_gemm.ptx`: compiler evidence.
- `0*_tests.log` and `0*.csv`: all lesson tests/benchmarks.

### Re-run on the prepared Pod

```bash
cd /workspace/gpu-kernel-lab
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export TRITON_CACHE_DIR=/workspace/.cache/triton
export HF_HUB_OFFLINE=1
python profiling/model_workload.py --output /workspace/baseline_repeat.json
python profiling/model_workload.py --variant triton_both --output /workspace/both_repeat.json
python profiling/quality_probe.py
KERNELLAB_STRICT_GEMM=1 python profiling/compare_native.py
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /workspace/decode_repeat \
  python profiling/model_workload.py --mode capture --phase decode --lengths 2048 --steps 4
```

Source, JSON/CSV/log evidence, generated GEMM PTX, and curated small raw Nsight
reports are backed up in GitHub. SQLite exports can be regenerated from raw
reports. Model weights and the virtual environment are under `/workspace`, not
in GitHub. A workspace path alone does not guarantee survival of Pod
termination; retain a separately persistent volume or re-download the pinned
model snapshot on a new Pod. No Pod stop/termination was performed by this run.
