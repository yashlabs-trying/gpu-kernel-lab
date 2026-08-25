# KernelLab Experiment Roadmap

This document is the working map for KernelLab. We will complete the experiments
incrementally, learn the relevant theory before implementing each one, and record
only results produced by reproducible benchmarks.

## Experimental rules

Every experiment follows the same engineering loop:

1. Explain the operation and its mathematics.
2. Identify tensor shapes, layouts, strides, and data types.
3. Predict whether the workload is memory-, compute-, launch-, or latency-bound.
4. Implement a clear PyTorch reference.
5. Test the reference across representative shapes.
6. Profile the reference implementation.
7. Implement the simplest correct Triton kernel.
8. Compare its output with the reference.
9. Benchmark with warmups and repeated measurements.
10. Change one important optimization variable at a time.
11. Explain why the change helped, hurt, or had no measurable effect.
12. Integrate useful kernels into the model and measure end-to-end impact.

We will never invent results or accept speedups that change the workload. Each
benchmark must record the GPU, software versions, data type, shapes, warmup,
repetition count, and summary statistics.

## Phase 0: Environment and reproducibility

### Experiment 0.1 — Remote GPU validation

- Verify the GPU model, compute capability, VRAM, driver, and CUDA versions.
- Verify PyTorch can allocate tensors and execute CUDA operations.
- Verify BF16 support.
- Compile and execute a minimal Triton kernel.
- Check access to Nsight Systems and Nsight Compute performance counters.
- Record the host CPU, RAM, disk, operating system, and container image.

### Experiment 0.2 — Reproducible environment

- Pin Python dependencies.
- Create environment-validation and setup scripts.
- Record PyTorch, Triton, Transformers, CUDA, and profiler versions.
- Make the environment reproducible on a newly rented GPU machine.
- Keep model weights and large traces outside Git.

### Experiment 0.3 — Benchmark harness validation

- Compare CPU wall-clock timing with CUDA event timing.
- Demonstrate why asynchronous CUDA work requires synchronization.
- Measure warmup and compilation effects.
- Report median, p20, and p80 latency instead of one execution.
- Detect and document unstable or thermally throttled runs.

## Phase 1: GPU execution fundamentals

### Experiment 1.1 — CPU versus GPU execution

- Run equivalent vector operations on the CPU and GPU.
- Vary tensor size to expose transfer cost and launch overhead.
- Explain when parallel GPU execution becomes worthwhile.

### Experiment 1.2 — Grid, programs, blocks, and masks

- Map Triton program IDs to tensor regions.
- Test power-of-two and irregular tensor sizes.
- Inspect boundary masks and deliberately reproduce an indexing bug.

### Experiment 1.3 — Vector addition

Implement and compare:

- PyTorch vector addition.
- Naive Triton vector addition.
- Tuned Triton vector addition.

Measure latency, effective bandwidth, and scaling across small and large tensors.
Study block size, coalescing, and launch-bound versus memory-bound behavior.

### Experiment 1.4 — Memory access patterns

- Compare contiguous, strided, and deliberately uncoalesced access.
- Compare aligned and misaligned offsets.
- Measure transpose-like and gather-like patterns.
- Relate requested bytes to achieved memory bandwidth.

## Phase 2: Elementwise kernels and fusion

### Experiment 2.1 — SiLU activation

- Implement PyTorch, naive Triton, and tuned Triton SiLU.
- Study sigmoid approximation cost and BF16/FP32 intermediates.
- Measure different tensor sizes and contiguous layouts.

### Experiment 2.2 — Fused SwiGLU

- Establish separate SiLU and multiply baselines.
- Implement `silu(gate) * up` in one Triton kernel.
- Count eliminated temporary tensors, reads, writes, and launches.
- Test model-relevant FFN shapes.

### Experiment 2.3 — General fusion limits

- Fuse increasingly long elementwise chains.
- Measure when fusion helps and when register pressure begins to hurt.
- Compare eager PyTorch, `torch.compile`, and explicit Triton fusion.

## Phase 3: Reductions and normalization

### Experiment 3.1 — Sum, maximum, and mean reductions

- Begin with one-row reductions.
- Compare reduction block sizes.
- Study FP32 accumulation for lower-precision inputs.
- Test rows smaller than, equal to, and larger than one Triton block.

### Experiment 3.2 — RMSNorm

- Implement the PyTorch reference equation.
- Implement naive and optimized Triton versions.
- Study epsilon, numerical stability, reduction strategy, and vector width.
- Test Qwen hidden sizes and representative batch/sequence combinations.

### Experiment 3.3 — Residual add plus RMSNorm

- Measure separate residual-add and normalization kernels.
- Fuse residual addition with RMSNorm.
- Quantify avoided HBM traffic and kernel launches.
- Verify both normalized output and updated residual values.

### Experiment 3.4 — Numerical precision study

- Compare FP32, FP16, and BF16 inputs and accumulation.
- Measure maximum absolute error, relative error, and mismatch distributions.
- Construct inputs that expose overflow, underflow, and cancellation.

## Phase 4: Layout-sensitive transformer operations

### Experiment 4.1 — Tensor layouts and strides

- Inspect contiguous and non-contiguous Q/K/V layouts.
- Compare reshape, view, transpose, and contiguous copies.
- Measure the cost of layout conversion independently.

### Experiment 4.2 — Rotary position embeddings

- Implement an explicit PyTorch RoPE reference.
- Implement naive and optimized Triton RoPE.
- Compare interleaved and split-half layouts where applicable.
- Test position offsets used during prefill and decode.

### Experiment 4.3 — Q/K RoPE fusion

- Apply RoPE to Q and K separately as a baseline.
- Evaluate a combined Q/K launch.
- Investigate projection-plus-RoPE fusion feasibility without replacing GEMM.
- Measure whether layout conversion erases the fusion benefit.

### Experiment 4.4 — KV-cache indexing and writes

- Implement token-by-token KV writes.
- Compare token-major, head-major, and block-oriented layouts.
- Measure contiguous and scattered access.
- Study grouped-query head mapping.

## Phase 5: Softmax

### Experiment 5.1 — Stable softmax

- Implement the max-subtracted PyTorch reference.
- Build naive and optimized row-wise Triton kernels.
- Test adversarial large and small logits.
- Study reduction width, exponentiation, and masking.

### Experiment 5.2 — Masked and causal softmax

- Add arbitrary masks and causal masks.
- Test short, irregular, and long rows.
- Quantify mask handling and padding overhead.

### Experiment 5.3 — Online softmax

- Implement the running maximum and normalization recurrence.
- Verify equivalence with materialized stable softmax.
- Use the experiment as preparation for blocked attention.

## Phase 6: Matrix multiplication

### Experiment 6.1 — Naive matrix multiplication

- Implement a pedagogical Triton GEMM.
- Explain the mapping from output tiles to program IDs.
- Measure redundant memory loads in the naive design.

### Experiment 6.2 — Tiled GEMM

- Tile M, N, and K dimensions.
- Accumulate in FP32 with BF16 inputs.
- Study reuse, Tensor Core eligibility, and boundary handling.

### Experiment 6.3 — GEMM configuration study

- Sweep block sizes and `num_warps`.
- Study occupancy, register pressure, and shared-memory pressure.
- Compare square, tall-skinny, and short-wide model shapes.

### Experiment 6.4 — Triton autotuning

- Define a controlled configuration search space.
- Cache selected configurations by shape.
- Measure autotuning cost separately from steady-state performance.

### Experiment 6.5 — Library comparison

- Compare educational GEMM with PyTorch/cuBLAS.
- Explain why matching a mature GEMM library is difficult.
- Keep library GEMMs in the integrated model unless custom work provides a
  reproducible advantage for an important shape.

## Phase 7: Attention

### Experiment 7.1 — Materialized attention reference

- Implement `softmax(QK^T / sqrt(d))V` explicitly.
- Record score-matrix size and memory traffic.
- Test causal and non-causal attention.

### Experiment 7.2 — Grouped-query attention

- Map query heads to shared key/value heads.
- Validate head broadcasting and indexing.
- Measure KV-memory savings relative to equal Q/K/V head counts.

### Experiment 7.3 — Blocked attention

- Tile Q, K, and V.
- Avoid loading the full working set at once.
- Study tile shape, causal boundaries, and sequence-length scaling.

### Experiment 7.4 — FlashAttention-style kernel

- Combine blocked QK computation, online softmax, and V accumulation.
- Avoid materializing the full attention score matrix.
- Track running maximum and normalization terms safely.
- Compare with PyTorch scaled dot-product attention.

### Experiment 7.5 — Prefill versus decode attention

- Benchmark multi-token prefill separately from single-token decode.
- Explain why their shapes and bottlenecks differ.
- Create distinct configuration choices where measurements justify them.

### Experiment 7.6 — Long-context scaling

- Sweep prompt length while holding other variables constant.
- Measure latency, memory use, and attention workspace growth.
- Identify the point at which each implementation becomes impractical.

## Phase 8: Real-model baseline

### Experiment 8.1 — Qwen3-0.6B loading and inspection

- Download and cache the official model through Hugging Face Transformers.
- Print architecture, parameter counts, layer structure, and tensor shapes.
- Trace each studied operation to its location in a transformer block.

### Experiment 8.2 — Correct generation baseline

- Run deterministic prompts in BF16.
- Separate tokenization, model loading, prefill, and decode time.
- Store prompts, seeds, and generation settings for repeatability.

### Experiment 8.3 — Model-level benchmark

- Test prompt lengths 128, 512, 2048, and 4096 where memory permits.
- Measure TTFT, prefill throughput, decode throughput, milliseconds per token,
  peak VRAM, and generated-token count.
- Keep batch size and generation settings fixed for before/after comparisons.

### Experiment 8.4 — PyTorch profiler baseline

- Capture CPU and CUDA activities.
- Rank kernels by total time, self time, and launch count.
- Identify synchronization gaps and expensive tensor transformations.

### Experiment 8.5 — Nsight Systems timeline

- Capture model startup separately from steady-state inference.
- Inspect CPU/GPU gaps, launches, synchronization, and memory operations.
- Create separate prefill and decode traces.

### Experiment 8.6 — Nsight Compute kernel analysis

- Inspect selected bottleneck kernels rather than profiling everything blindly.
- Record achieved bandwidth, compute utilization, occupancy, registers, cache
  behavior, and warp-level metrics.
- Compare measured behavior with the predicted bottleneck.

## Phase 9: Model integration

### Experiment 9.1 — Kernel replacement mechanism

- Build explicit, reversible patches for selected Qwen operations.
- Preserve a switch between baseline and optimized execution.
- Fail clearly when an unsupported shape or data type is encountered.

### Experiment 9.2 — Integrate fused SwiGLU

- Replace only the activation/multiply portion around optimized GEMMs.
- Run layer-level and end-to-end correctness tests.
- Measure whether the model-level effect survives framework overhead.

### Experiment 9.3 — Integrate RMSNorm variants

- Replace RMSNorm and residual-plus-RMSNorm where architecture permits.
- Measure launch-count, memory-traffic, and latency changes.

### Experiment 9.4 — Integrate RoPE and KV operations

- Patch Q/K RoPE and selected cache operations.
- Test both prefill and incremental decode positions.
- Verify cache contents against the reference model.

### Experiment 9.5 — Integrate attention

- Add a guarded custom-attention path for supported configurations.
- Compare against PyTorch SDPA for correctness and performance.
- Fall back safely for unsupported shapes.

### Experiment 9.6 — Cumulative optimization study

- Enable optimized kernels individually and cumulatively.
- Identify interactions, regressions, and double-counted improvements.
- Keep only changes with reproducible end-to-end value.

## Phase 10: Advanced kernel investigations

### Experiment 10.1 — Roofline analysis

- Calculate operational intensity for representative kernels.
- Compare theoretical bandwidth/compute ceilings with measured throughput.
- Place memory-bound and compute-bound kernels on a roofline chart.

### Experiment 10.2 — Occupancy and register pressure

- Vary block shape and warp count while recording register usage.
- Demonstrate a case where more parallel work reduces occupancy.
- Separate occupancy from useful utilization.

### Experiment 10.3 — Persistent and grouped scheduling

- Explore persistent execution for suitable workloads.
- Compare row-major and grouped program ordering for GEMM locality.
- Document hardware- and shape-specific tradeoffs.

### Experiment 10.4 — Mixed precision and FP8 exploration

- Establish BF16 as the correctness baseline.
- Explore FP16 and supported FP8 paths without changing comparison rules.
- Measure accuracy, conversion overhead, throughput, and hardware dependence.

### Experiment 10.5 — CUDA Graphs and launch overhead

- Measure decode launch overhead.
- Capture a stable decode path where feasible.
- Separate graph benefits from custom-kernel benefits.

### Experiment 10.6 — `torch.compile` comparison

- Compare eager PyTorch, compiled PyTorch, and explicit Triton kernels.
- Inspect which operations the compiler already fuses.
- Focus custom engineering on gaps that remain measurable.

## Phase 11: Advanced inference project

The inference project will use a 7–8B-class model after the small-model kernel
work establishes a correct methodology.

### Experiment 11.1 — 7–8B model feasibility

- Record BF16 weight memory, runtime allocations, and KV-cache growth.
- Determine safe context and batch sizes for the selected GPU.
- Compare BF16 and weight-only quantized loading where useful.

### Experiment 11.2 — Prefill workload study

- Sweep prompt length and batch size.
- Measure TTFT, prompt tokens/second, GEMM utilization, and peak VRAM.
- Identify compute- and memory-limited regions.

### Experiment 11.3 — Decode workload study

- Sweep KV-cache length and batch size.
- Measure tokens/second, milliseconds/token, bandwidth, and launch overhead.
- Identify operations that dominate single-token generation.

### Experiment 11.4 — Continuous and dynamic batching concepts

- Build a small educational batching experiment rather than a production server.
- Measure throughput/latency tradeoffs across concurrent sequences.
- Explain padding waste and scheduling effects.

### Experiment 11.5 — KV-cache memory experiments

- Calculate cache bytes per token and per sequence.
- Compare data types and layouts.
- Explore paged-cache concepts and fragmentation through controlled examples.

### Experiment 11.6 — Quantization comparison

- Compare BF16 with selected 8-bit and 4-bit weight formats.
- Measure load time, VRAM, prefill, decode, and output-quality differences.
- Avoid mixing quantization speedups into kernel-only claims.

### Experiment 11.7 — Serving-engine comparison

- Compare the educational PyTorch path with one established inference engine.
- Hold model, prompts, output length, and precision as constant as possible.
- Explain optimizations the production engine applies beyond our project scope.

### Experiment 11.8 — End-to-end inference report

- Report throughput, TTFT, inter-token latency, peak VRAM, and cost estimates.
- Separate single-user latency from multi-request throughput.
- Document limitations and the next bottleneck rather than claiming universal
  performance.

## Phase 12: Cross-GPU and reproducibility studies

### Experiment 12.1 — Architecture comparison

- Repeat selected kernels on another NVIDIA architecture when budget permits.
- Compare memory bandwidth, Tensor Core behavior, and preferred tile sizes.
- Never combine numbers from different machines in one speedup claim.

### Experiment 12.2 — Performance stability

- Repeat important benchmarks across sessions.
- Record clock, power, temperature, and host-sharing effects when visible.
- Attach uncertainty to small performance differences.

### Experiment 12.3 — Cost efficiency

- Record active GPU time and marketplace rate.
- Compare useful experiments completed per dollar.
- Separate storage and network charges from compute charges.

## Final before/after evaluation

The final evaluation will run the exact same prompts, shapes, data types,
generation settings, and correctness thresholds for both configurations:

```text
Baseline Qwen
    versus
Qwen with selected KernelLab kernels
```

It will report:

- TTFT.
- Prefill tokens per second.
- Decode tokens per second and milliseconds per token.
- Peak VRAM.
- Kernel launches.
- Important kernel latency and achieved bandwidth/compute.
- Numerical error and generation-level correctness checks.
- Speedup or regression for every tested workload.

## Definition of done

KernelLab is complete only when:

- Every selected kernel has a reference, tests, benchmarks, and explanation.
- Profiling evidence supports each bottleneck claim.
- All integrated kernels have safe fallbacks.
- End-to-end measurements use an unchanged workload.
- Results include unsuccessful experiments and limitations.
- The environment and commands are reproducible from a fresh GPU instance.
- No performance number is included unless it was actually measured.

