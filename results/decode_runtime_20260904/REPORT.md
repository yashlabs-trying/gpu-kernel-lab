# Decode runtime: residual fusion, phase-specific MLP, and CUDA Graphs

Date: 2026-09-04  
GPU: RTX 3090, BF16, batch 1  
Model: `Qwen/Qwen3-0.6B`  
Context: 2,048 tokens; 16 measured greedy decode steps

## Result

| Path | Median latency | Serial throughput |
|---|---:|---:|
| Dynamic-cache hybrid baseline | 20.203 ms/token | 49.50 token/s |
| Fused static/split-KV + residual norm | 15.532 ms/token | 64.38 token/s |
| Same path replayed as a CUDA Graph | **2.256 ms/token** | **443.17 token/s** |

The non-graph integration is 23.12% lower latency than its matched dynamic-cache
baseline. Graph replay is 85.47% lower than the same eager optimized path and
88.83% lower than the dynamic baseline in this narrow benchmark.

These are single-request, batch-1, greedy, GPU-event measurements. They exclude
prefill, model loading, cache construction, tokenization, detokenization, network
transport, EOS handling, and sampling/logits processors. They are not a claim of
443 token/s from a production server.

## Launch structure

Nsight Systems node tracing recorded 4,560 kernels over 16 graph replays:
**285 GPU kernels/token**. This reaches the first 300-400 target. CUDA Graphs do
not fuse these kernels; they replace hundreds of CPU launch/dispatcher operations
with one graph replay submission per token.

The traced GPU span was 36.754 ms (2.297 ms/token), consistent with the unprofiled
2.256 ms/token result. Kernel-busy union was 36.515 ms, or 99.35% of the traced
span. That is timeline utilization, not achieved SM occupancy.

Principal nodes per token:

| Work | Kernels/token |
|---|---:|
| cuBLAS GEMV group | 56 |
| W8 dual gate/up GEMV + SwiGLU | 28 |
| split-KV partial | 28 |
| BF16 projection kernel | 28 |
| Q/K norm + RoPE + cache/mask write | 28 |
| split-KV merge | 28 |
| fused residual + post-attention RMSNorm | 28 |
| RMSNorm | 29 |
| remaining MLP residual add | 28 |
| LM-head GEMV, argmax, token select, position advance | 4 total |

The next safe launch-reduction targets are therefore the remaining MLP residual
boundary and coordinated projection epilogues. They should only be integrated if
their end-to-end latency beats the graph baseline.

## Changes

### Residual + RMSNorm

Each decoder layer now fuses:

```text
attention output + residual -> saved residual + post-attention RMSNorm
```

The kernel performs the addition and FP32 reduction without materializing an
extra normalization input. The final MLP residual remains separate because
fusing across the next layer changes the decoder-layer ABI.

### Phase-specific MLP

One kernel no longer serves both phases:

- Prefill (`M > 1`): tiled BF16 tensor-core gate/up projection with SwiGLU
  epilogue; contexts above the validated 4,096-row range fall back safely.
- Decode (`M = 1`): calibrated W8A16 dual GEMV. Independent warp reductions
  stream packed gate/up weights and write only the SwiGLU output.

The dense Triton dispatcher also now recognizes `[1,1,K]` as `M=1`; previously
it selected GEMM merely because the tensor was not one-dimensional.

### CUDA Graph replay

The graph owns fixed-address token and position buffers and reuses the existing
fixed cache, mask, RoPE tables, and split-attention workspaces. Argmax, next-token
copy, cache write, mask update, and position increment are captured.

## Correctness

- 32 targeted kernel/static-cache/runtime tests passed.
- A 16-step eager-versus-graph check had identical tokens, final position, and
  stable token/mask/position addresses.
- The 247-token teacher-forced probe retained 98.38% argmax agreement with the
  dynamic hybrid reference and the same NLL values as the previously validated
  hybrid split-KV path.

The teacher-forced probe is small and is not a standard model-quality benchmark.

## Constraints

- Eager Qwen3-0.6B, BF16, batch 1, one-token greedy decode only.
- No sampling, logits processors, variable batches, EOS stopping, concurrent
  requests, or production `generate()` cache adapter inside the graph.
- Graph construction consumes one warmup step; capacity accounts for it.
- Weight/calibration storage remains process-local and dense weights are retained.
- Different context buckets or batch shapes require separate graph instances.

## Reproduce

```bash
pytest -q learning/12_decode_runtime/test_decode_runtime.py \
  learning/10_cuda_decode/test_split_kv.py learning/11_static_kv/test_static_kv.py
python profiling/decode_graph_check.py
python profiling/static_kv_quality.py --hybrid --split-attention --residual-norm \
  --output results/decode_runtime_20260904/quality.json
python profiling/static_kv_benchmark.py --lengths 2048 --hybrid \
  --split-attention --residual-norm --cuda-graph \
  --output results/decode_runtime_20260904/graph.json
```

Raw JSON, logs, and the Nsight `.nsys-rep` capture are stored beside this report.
