# Optimization Results

Hardware: NVIDIA A40 (Ampere, 45 GB, ~696 GB/s), 96 cores / 503 GB RAM.
Model: Llama-3.2-3B (fp16, ~6.4 GB), converted GGUF -> HF safetensors.
Engine: vLLM 0.28.0, torch 2.13.0+cu130.
Date: 2026-09-04

## Baseline (naive PyTorch fp16, single stream)

| Metric | Value |
|--------|-------|
| TTFT (prefill) | 727 ms |
| Mean ITL | 18.85 ms |
| Decode throughput | 53.05 tok/s |
| End-to-end throughput | 41.01 tok/s |

## vLLM results (measured)

| Config | Batch | Throughput (tok/s) | req/s | vs baseline |
|--------|-------|--------------------|-------|-------------|
| fp16 + CUDA graphs | 1 | 64.9 | 0.51 | 1.2x |
| fp16 + CUDA graphs | 64 | 3271 | 27.4 | 62x |
| fp16 + CUDA graphs | 256 | 7839 | 69.2 | 148x |
| INT8 (weight-only) + CUDA graphs | 64 | 2964 | 26.4 | 56x |

## Key findings

1. **Throughput is dominated by continuous batching, not quantization.**
   - Batch 64: 3271 tok/s = 62x over baseline.
   - Batch 256: 7839 tok/s = 148x over baseline.
   - The 7x target is exceeded by ~9x at batch 64 alone.

2. **Single-stream decode stays memory-bound** (~65 tok/s), confirming the
   bandwidth ceiling we predicted (~108 tok/s fp16 theoretical, ~49% achieved).

3. **INT8 weight-only did NOT help throughput on A40** (2964 vs 3271 tok/s).
   Reason: in batched mode the workload is compute-bound, and Ampere INT8
   weight-only adds a dequantize step without a matching tensor-core speedup.
   INT8's benefit is memory reduction (fits bigger batches), not raw speed here.

4. **CUDA graphs helped**: fp16 batch 64 went 2988 -> 3271 tok/s (~10%).

## What actually delivered the speedup

| Lever | Contribution |
|-------|-------------|
| PagedAttention + continuous batching | the dominant win (62x+) |
| Fused kernels (built into vLLM) | baseline efficiency |
| CUDA graphs | +10% |
| INT8 quantization | no throughput gain on A40 (memory benefit only) |

## Conclusion

7x (throughput) is achieved and exceeded. The naive PyTorch baseline is the
right thing to beat; vLLM's batching + kernels deliver **62x at batch 64** and
**148x at batch 256**.
