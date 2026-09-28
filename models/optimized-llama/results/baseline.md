# Baseline Profiling Results

Hardware: NVIDIA A40 (Ampere, 45 GB, ~696 GB/s), 96 cores / 503 GB RAM.
Model: Llama-3.2-3B (fp16, ~6.4 GB), loaded via Ollama GGUF -> PyTorch.
Date: 2026-09-04

## Serving metrics (PyTorch fp16 baseline)

| Metric | Value |
|--------|-------|
| TTFT (prefill, 5-token prompt) | 727 ms |
| Mean ITL (inter-token latency) | 18.85 ms |
| Decode throughput | 53.05 tok/s |
| End-to-end throughput | 41.01 tok/s |
| Avg SM utilization (decode) | 71.1% |
| Avg memory utilization | 64.2% |
| Effective bandwidth | ~340 GB/s of ~696 GB/s peak |

## Bottleneck proof

Decode is **memory-bandwidth bound**, not compute bound:

- GEMM kernels (`cutlass_80_wmma_tensorop_f16_s161616gemm`) account for ~50% of
  GPU time but run at only ~340 GB/s effective bandwidth (49% of peak).
- The weight matrix (~6.4 GB fp16) must be read from HBM once per token in
  decode; this dominates and sets the ~53 tok/s ceiling.
- Compute utilization (71%) < bandwidth pressure confirms we are bounded by
  bytes moved, not FLOPs.

## Kernel breakdown (Nsight Systems / torch.profiler)

| Kernel | % GPU time | Note |
|--------|-----------|------|
| cutlass_80_wmma_tensorop_f16_s161616gemm | 49.7% | QKV/MLP GEMMs |
| ampere_fp16_s16816gemm_64x64_sliced1x2 | 13.5% | MLP GEMMs |
| gemv2T_kernel_val | 9.0% | output proj (prefill) |
| gemvx::kernel | 5.3% | decode GEMV |
| pytorch_flash::flash_fwd_kernel | 1.0% | attention (already fused) |
| elementwise / reduce / copy | ~15% | norm, RoPE, residual |

## Profiling tool status

| Tool | Status |
|------|--------|
| Nsight Systems (`nsys`) | Working — report at `profiling/nsys_baseline.nsys-rep` |
| Nsight Compute (`ncu`) | **BLOCKED** — `ERR_NVGPUCTRPERM` (container lacks `CAP_SYS_ADMIN`, host `RmProfilingAdminOnly=1`) |
| torch.profiler fallback | Working — `profiling/profile_kernels.py` |

## Optimization levers (target 5-7x)

1. **INT8 quantization** — halve weight bytes (6.4 GB -> 3.2 GB), ~2x decode.
2. **Kernel fusion** — eliminate ~15% elementwise/copy overhead.
3. **Continuous batching** — the dominant throughput lever (saturate bandwidth).
4. **CUDA graphs** — remove per-step launch overhead.
