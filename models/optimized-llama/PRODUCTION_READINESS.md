# Llama production-readiness tracker

Last updated: 2026-10-01

Legend: **implemented** means the repository enforces the method; **validated**
requires a fresh GPU result produced by schema version 2.

## Phase 1 — benchmark methodology

Implemented in schema-v2 runners:

- [x] Real TTFT from first-token receipt; generation averages are not TTFT.
- [x] Real per-request ITL from consecutive token timestamps.
- [x] Separate input-token and output-token throughput.
- [x] Prompt tokens excluded from decode throughput.
- [x] High-concurrency throughput is not labeled as a batch-1 speedup.
- [x] Real vLLM streaming timestamps; no zero-filled latency fields.
- [x] Warmups before measured runs.
- [x] CUDA synchronization around PyTorch wall timing.
- [x] CUDA events for GPU-only latency.
- [x] Monotonic request/server-client timing.
- [x] Deterministic prompt token IDs and seeds.
- [x] Raw request and token samples retained.
- [x] Default 100 repetitions and no p99 below 100 successful requests.
- [x] Median, p90, p95, and p99 summaries.
- [x] Software, CUDA, GPU, clock, P-state, power, and memory metadata.
- [x] Model-load and first-inference timing separated from warm measurements.

Still requiring GPU execution:

- [x] Produce fresh matched PyTorch batch-1 versus vLLM concurrency-1 results.
- [x] Add a true concurrent PyTorch runner for concurrency-N comparisons (GPU validation pending).
- [x] Run matched vLLM concurrency 1, 2, 4, 8, 16, 32, and 64.
- [x] Run CUDA Graph enabled versus `--enforce-eager`.
- [x] Run FP16 versus INT8 with identical prompts, output lengths, and concurrency
  (INT8 performance passed, but its quality release gate failed).
- [ ] Validate schema-v2 output on both supported CUDA environments.
- [x] Add process/model cold-start benchmark separate from first-request warmup.
- [x] Execute and publish the process/model cold-start artifact.

No historical schema-v1 number passes Phase 1. Phase 2 acceptance execution
begins only after one fresh matched schema-v2 pair is recorded and reviewed.

## Phase 2 — quality tooling

- [x] Deterministic FP16/candidate capture format for logits and generated IDs.
- [x] Per-layer hidden-state cosine and maximum-difference comparison.
- [x] Argmax agreement, token agreement, NLL, and perplexity metrics.
- [x] Short/medium/long factual, reasoning, code, and multilingual prompt corpus.
- [x] Tokenizer parity and model checksum tooling.
- [x] Release gates encode ≥99% argmax and ≥0.999 minimum cosine.
- [x] Execute converted-HF FP16 and INT8 capture gates (INT8 was correctly rejected).
- [x] Complete an 8,192-token serving gate and a 1,000-request stability run.
- [x] Execute and publish the full multi-domain quality prompt corpus (FP16
  passed as reference; BitsAndBytes INT8 was correctly rejected).
- [x] Add a pinned, deterministic LM Evaluation Harness runner with raw sample logging.
- [x] Execute the full standard task suite and publish its result artifact.

## Phase 3 — original Llama kernels

- [x] Static shape exporter records every layer and projection dimension from config.
- [x] Kernel provenance explicitly separates local work from vLLM/framework kernels.
- [x] Implement standalone and fused-residual Llama RMSNorm candidates plus a
  direct active-vLLM CUDA-operation acceptance benchmark.
- [x] Capture runtime hot shapes and active framework/vLLM kernels on GPU.
- [x] Implement, validate, and benchmark Llama RMSNorm candidates on GPU (both
  candidates were slower than vLLM and were correctly rejected).
- [ ] Accept only candidates that beat the active vLLM path end to end.

## Phase 4 — serving and load harness

- [x] Real OpenAI-compatible streaming client with token-ID timestamps.
- [x] Matched concurrency and input/output-length matrix generator.
- [x] Error rate, request rate, separate token rates, raw samples, and environment data.
- [x] CLI exposes deterministic PyTorch and vLLM benchmark paths.
- [x] Cancellation, soak, and overload probes are implemented.
- [x] Mixed short/long scheduler-fairness benchmark is implemented.
- [x] Frontier, saturation, fairness, and scaling analysis is implemented.
- [x] Execute concurrency, saturation, cancellation, and overload runs.
- [x] Execute mixed-load length fairness and the formal one-hour soak.

## Phase 5 — local usability

- [x] Installable `pyproject.toml` and `kernellab` CLI.
- [x] Model registry plus download, serve, benchmark, doctor, plan, and validation commands.
- [x] Exact CUDA 12.8 and CUDA 13.0 dependency manifests.
- [x] CUDA 12.8/13.0 Dockerfiles and NVIDIA Container Toolkit deployment guide.
- [x] Native Hugging Face weights by default; GGUF conversion remains optional.
- [x] Checksum and tokenizer-parity verification.
- [x] Context length, GPU-memory utilization, tensor parallelism, and eager/graph controls.
- [x] vLLM OpenAI API supplies health, metrics, models, completions, chat, streaming, and graceful shutdown.
- [ ] Build container and smoke-test model loading/OOM messages on GPU.

## Shared repository readiness

- [x] Root license, package metadata, CLI entry point, model registry, and contribution guide.
- [x] CPU GitHub Actions, formatting/static-analysis configuration, and test markers.
- [x] Benchmark JSON schema and automated semantic validation.
- [x] Architecture diagram, changelog, and reproducibility matrix.
- [ ] Add a GPU CI runner when credentials and hardware are available.
- [ ] Publish a tagged release only after GPU acceptance artifacts pass.

## Phase 6 — portability and multi-GPU

- [x] Runtime metadata records attention backend label, CUDA Graph mode, dtype,
  quantization, tensor-parallel size, GPU identity, and NVIDIA topology.
- [x] Serve CLI exposes tensor-parallel sizes, attention backend, context, memory,
  sequence capacity, eager/graph, and chunked-prefill controls.
- [x] Analysis computes tensor-parallel speedup and scaling efficiency.
- [ ] Validate Ampere and a second GPU architecture.
- [ ] Compare FlashAttention/FlashInfer under matched settings.
- [ ] Execute TP1/TP2 and TP4 when available; validate logits against TP1.
- [ ] Measure PCIe/NVLink communication overhead and worker-failure recovery.
