# Llama kernel provenance

No Llama-specific custom kernel is currently accepted in this repository.
Production execution uses kernels selected by vLLM, PyTorch, cuBLAS/CUTLASS,
the configured attention backend, and their transitive CUDA dependencies.
Installing or configuring vLLM is not counted as original kernel engineering.

The first local Llama candidate is
`kernels/llama_rmsnorm.py`: standalone RMSNorm and residual-add + RMSNorm for
contiguous FP16/BF16 inference rows. Its acceptance runner
`kernels/bench_rmsnorm_vs_vllm.py` uses the installed vLLM `_C.rms_norm` and
`_C.fused_add_rms_norm` operations as the baseline, records 100 raw CUDA-event
samples per case, and marks candidates eligible for end-to-end validation only
when correctness and matched median latency pass. A microbenchmark winner is
not an accepted production path.

The local repository also supplies benchmark methodology, quality gates, shape
export, profiling targets, serving/load orchestration, and deployment
infrastructure. Each local RMSNorm, residual fusion, RoPE/KV write, GQA decode
attention, SwiGLU, projection, or GEMV candidate must:

1. be labeled with its exact supported Llama shapes and architecture;
2. pass FP16 token/logit/layer quality gates;
3. be compared with the active vLLM kernel on the same environment;
4. improve the complete matched workload, not only naive PyTorch; and
5. retain a safe framework fallback.

Candidates that lose to vLLM or miss correctness gates remain experiments and
are not enabled by default. The generated shape manifest from
`profiling/export_shapes.py` is the source of dimensions for future candidates.
