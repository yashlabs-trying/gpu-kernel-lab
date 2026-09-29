# Optimized model portfolio

GPU Kernel Lab is growing into a four-model inference optimization portfolio.
Each model project keeps its own environment, baselines, profilers, measurements,
and architecture-specific notes while sharing the laboratory's acceptance rules:

1. Measure a native framework baseline on the same GPU and software stack.
2. Separate prefill, decode, and concurrent-serving results.
3. Validate numerical or task-level quality before accepting an optimization.
4. Report end-to-end performance; do not promote isolated kernel wins as model wins.
5. Preserve a safe library fallback for unsupported shapes and architectures.

## Portfolio

| Slot | Model project | Architecture | Status |
|---:|---|---|---|
| 1 | [Qwen3-0.6B](qwen3-0.6b/README.md) | Qwen3 | Exact single-GPU serving path accepted on SM86/SM89; portable fallback and real concurrent load tested |
| 2 | [Llama-3.2-3B](optimized-llama/README.md) | Llama | Imported optimized vLLM benchmark and profiling project |
| 3 | To be selected | — | Planned |
| 4 | To be selected | — | Planned |

The Llama project is a pinned in-tree snapshot, not a Git submodule. A normal
clone therefore contains all benchmark and profiling code. Its upstream source
and synchronization procedure are recorded in
[`optimized-llama/UPSTREAM.md`](optimized-llama/UPSTREAM.md).

Model weights are never committed. Store them on a persistent GPU volume and
follow each model project's setup instructions.
