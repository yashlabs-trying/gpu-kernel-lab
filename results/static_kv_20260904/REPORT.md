# Integrated static KV-cache experiment

Date: 2026-09-04  
GPU: NVIDIA GeForce RTX 3090 (24 GiB, compute capability 8.6)  
Model: `Qwen/Qwen3-0.6B`, BF16, batch size 1, greedy next-token selection

## Outcome

The previously isolated `Q RMSNorm + K RMSNorm -> RoPE -> direct K/V write`
kernel is now integrated into an eager Qwen decode path. The path uses fixed-address,
layer-major K and V buffers and can optionally replace PyTorch SDPA with a
preallocated split-KV decode-attention implementation.

The strongest matched comparison at a 2,048-token context combines the earlier
projection/SwiGLU work with the new fused cache and split-KV attention:

| Path | Median latency | Serial throughput |
|---|---:|---:|
| Dynamic cache + earlier hybrid kernels | 20.631 ms/token | 48.47 token/s |
| Hybrid + fused static cache + split-KV | 15.668 ms/token | 63.82 token/s |
| Change | **24.06% lower latency** | **31.68% higher throughput** |

This is a model-level result, unlike the earlier isolated 25x microbenchmark.
That 25x result must not be interpreted as a 25x model speedup.

A separate plain-Transformers harness measured 27.863 ms/token. Compared with
15.668 ms/token, the contextual end-to-end reduction is 43.77% and inverse-latency
throughput rises 77.83%. This is encouraging but is **not** as rigorous as the
matched 24.06% comparison: it came from a separate harness/process, and its two
runs shifted from 27.009 to 28.498 ms/token.

## What was integrated

The decode cache is physically allocated as contiguous BF16 arrays:

```text
key_storage   [layer, batch, kv_head, capacity, head_dim]
value_storage [layer, batch, kv_head, capacity, head_dim]
```

Each layer receives a stable slice of these arrays. The single-token path now:

```text
Q/K/V projections
  -> Q and K RMSNorm with FP32 reductions
  -> Q and K RoPE using precomputed compact sin/cos tables
  -> direct K and V write at the current cache position
  -> open the persistent mask position in the same Triton launch
  -> attention over valid cache positions
  -> advance a persistent GPU position scalar
```

The optional split-KV implementation accepts the persistent valid-position scalar
and preallocated FP32 partial/LSE workspaces. It avoids GQA head expansion and does
not allocate per-token attention intermediates.

## Matched latency results

All numbers below are unprofiled medians, with two warmups, five repeats, 16 decode
steps, and cache allocation/prefill excluded.

### Fused RMSNorm/RoPE/cache write with PyTorch SDPA

| Context | Dynamic cache | Fused static | Samples | Latency reduction |
|---:|---:|---:|---:|---:|
| 128 | 23.159 ms | 19.519 ms | 5 | 15.72% |
| 512 | 23.438 ms | 18.660 ms | 5 | 20.39% |
| 2,048 | 22.922 ms | 19.130 ms | 10 | 16.54% |
| 4,096 | 23.543 ms | 18.444 ms | 5 | 21.66% |

Using Transformers `StaticCache` alone was slower than its dynamic baseline by
6.06-9.68% in this setup. Fixed storage by itself is not the optimization; the
fused writes and removal of surrounding work are essential.

### Split-KV decode attention

| Context | Dynamic cache | Fused static + split-KV | Samples | Latency reduction |
|---:|---:|---:|---:|---:|
| 2,048 | 22.826 ms | 18.149 ms | 10 | 20.49% |
| 4,096 | 22.708 ms | 18.056 ms | 5 | 20.49% |

### Composition with earlier projection/SwiGLU kernels (context 2,048)

| Variant | Dynamic | Fused static | Samples | Latency reduction |
|---|---:|---:|---:|---:|
| Hybrid + PyTorch SDPA | 20.502 ms | 16.227 ms | 10 | 20.85% |
| Hybrid + split-KV | 20.631 ms | **15.668 ms** | 10 | **24.06%** |

## Nsight Systems evidence

The trace covers four tokens at context 2,048. Profiler timings include overhead,
so they diagnose launch structure and bottlenecks; unprofiled measurements above
are the speed claims.

| Path | Kernels | Kernels/token | Kernel-time sum | GPU timeline span |
|---|---:|---:|---:|---:|
| Dynamic cache | 3,332 | 833 | 18.481 ms | 130.886 ms |
| Transformers static cache | 3,868 | 967 | 33.320 ms | 141.350 ms |
| Fused static + PyTorch SDPA | 2,036 | 509 | 29.584 ms | 106.456 ms |
| Fused static + split-KV | **1,812** | **453** | **11.326 ms** | **92.182 ms** |

Relative to dynamic cache, the final path launches 380 fewer kernels per token,
a 45.62% reduction. The dynamic trace contained 3.439 ms of concatenation across
four tokens. The fused-static SDPA trace eliminated that category but still spent
4.382 ms in 224 direct-copy kernels, primarily GQA expansion. Split-KV eliminated
both the concatenation and copy/cast categories.

In the final trace each of 28 layers launches one split-attention partial kernel
and one merge kernel per token. Across four tokens those cost 1.694 ms and
0.229 ms respectively. The fused Q/K norm, RoPE, cache write, and mask update was
112 launches totaling 0.225 ms.

The trace's 12.3% kernel-busy fraction is a timeline utilization measure, not SM
occupancy. It shows that Python/framework launch gaps remain large; it must not be
reported as achieved hardware occupancy.

## Numerical checks

The final integration suite passed **47 tests**. A small 247-token teacher-forced
check produced:

| Path | Mean NLL | Argmax agreement vs dynamic |
|---|---:|---:|
| Dynamic hybrid cache | 3.360715 | reference |
| Transformers static cache | 3.362674 | 98.79% |
| Hybrid + fused static + split-KV | 3.355727 | 98.38% |

These results check implementation fidelity on the selected passages; they are
not a substitute for a standard model-quality evaluation. BF16 reordering can
change near-tied argmax decisions, and the small lower NLL is not evidence of a
general quality improvement.

## Memory cost

K plus V allocation is:

```text
2 * layers * batch * kv_heads * capacity * head_dim * bytes_per_element
```

For this model (`28 * 1 * 8 * capacity * 128`, BF16), this is 114,688 bytes per
cache position: 225.75 MiB at capacity 2,064; 449.75 MiB at 4,112; and 4.375 GiB
at the model maximum of 40,960. Benchmarks allocate only `context + 16`, not the
full maximum.

## Remaining work and limitations

- CUDA Graph capture is not implemented yet. Cache, mask, position, and attention
  workspace addresses are stable; stable token/logit buffers and full decode-step
  capture still need implementation.
- The experimental adapter targets eager, BF16, batch-1 Qwen3-0.6B geometry. It is
  not yet a general Transformers cache implementation.
- It currently requires explicit positions/masks, does not support concurrent
  requests, and intentionally leaves Transformers cache-length metadata at the
  prefill length. `generate()` compatibility needs a proper cache adapter.
- Projection GEMV dominates the final profile. The two largest remaining groups
  are 560 cuBLAS GEMV launches (3.825 ms) and 224 BF16 GEMM launches (2.886 ms)
  across the four traced tokens.
- Residual adds, RMSNorm, SiLU, and multiply remain separate in parts of the model.
- The split-KV path is an experimental correctness/performance candidate, not yet
  a production replacement for optimized serving runtimes.

## Reproduction

```bash
pytest -q learning/10_cuda_decode/test_split_kv.py learning/11_static_kv/test_static_kv.py
python profiling/static_kv_quality.py --hybrid --split-attention
python profiling/static_kv_benchmark.py --context 2048 --hybrid --split-attention
python profiling/summarize_static_kv.py results/static_kv_20260904
```

Raw JSON, logs, and `.nsys-rep` captures are stored beside this report.
