# Changelog

## 0.2.0 — unreleased

- Replaced invalid Llama latency and throughput methodology with schema-v2 raw
  streaming timestamps, synchronized CUDA events, and matched workloads.
- Added the `kernellab` model registry, download/serve/benchmark commands,
  hardware detection, benchmark-matrix planning, and result validation.
- Added deterministic Llama quality capture and token/logit/layer/NLL gates.
- Added multi-domain corpus capture, pinned LM Evaluation Harness integration,
  mixed-length scheduler fairness, process cold-start timing, and profiler traces.
- Added local Llama RMSNorm/fused-residual candidates with direct active-vLLM
  CUDA-operation comparison; production acceptance remains end-to-end gated.
- Added exact environment manifests, CUDA 12.8/13.0 Dockerfiles, CPU CI,
  architecture/deployment documentation, and repository license.
- Marked historical unmatched Llama speedups as invalid for production use.
