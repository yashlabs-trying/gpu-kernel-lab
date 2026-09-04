# Serving acceptance report — 2026-09-04

## Environment

- GPU host: RunPod, NVIDIA GeForce RTX 3090 (24 GB, compute capability 8.6)
- Python environment: `/workspace/gpu-kernel-lab-final/.venv`
- PyTorch 2.8.0+cu128, Transformers 5.16.1, Triton 3.4.0
- Clean checkout: `/workspace/gpu-kernel-lab-final`

## Test results

- Full repository suite: **476 passed** in 210.76 s.
- Final serving suite: **10 passed** in 3.46 s.
- Remaining warning is a third-party Starlette `TestClient`/httpx deprecation warning.

The serving tests cover paged-KV allocation/reuse, admission capacity, continuous batching,
chunked prefill, cancellation, timeouts, sampling, metrics, OpenAI-compatible JSON and
streaming responses, validation errors, and request-capacity failures.

## 5,000-request control-plane stress test

- Submitted: **5,000**
- Completed: **4,864**
- Intentionally cancelled: **136**
- Failed / timed out / rejected: **0 / 0 / 0**
- Terminal event present for every request: **yes**
- KV blocks after run: **60,000 / 60,000 free**
- Reserved or leaked blocks: **0**
- Wall time: **25.277 s**
- Control-plane rate: **197.81 requests/s**
- Fake-executor generated-token rate: **1,646.95 tokens/s**

The throughput and latency values from this stress test measure scheduler/API behavior with
the deterministic `FakeExecutor`. They are **not** Qwen GPU inference performance and must not
be used as model-serving claims. The long TTFT values reflect queueing 5,000 simultaneous
requests through a deliberately small synthetic execution budget.

## Existing GPU profiling baseline

The preceding RTX 3090 profiling campaign remains recorded in
`results/final_newpod_20260904/REPORT.md`: the experimental graph path reached 2.253 ms ITL at
context 2,048, 443.9 tokens/s in its serial GPU-only benchmark, and 258 kernels/token. Nsight
Systems profiling completed. Nsight Compute hardware-counter collection was blocked by the
host's `ERR_NVGPUCTRPERM` permission setting.

## Acceptance verdict

The new **serving control plane passes its functional and stress gates**. It has deterministic
cache reclamation, lifecycle handling, streaming API behavior, and observable metrics.

The complete system is **not yet production-ready** for Qwen traffic. Two release blockers
remain:

1. Connect the paged-KV scheduler to a real GPU Qwen executor/paged-attention kernel. Current
   acceptance tests use a fake executor, while the existing optimized model path still uses its
   separate contiguous/static-cache experiments.
2. Pass the model-quality gate. Earlier optimized-path evaluation measured 98.05% RMS agreement
   and 96.48% full-path agreement, below the required production correctness threshold.

Only after those blockers pass should the API undergo real multi-client GPU soak tests and be
reported with production TTFT, ITL, throughput, and tail-latency numbers.
