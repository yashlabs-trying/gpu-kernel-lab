# GPU Kernel Lab

Custom Triton kernels and model-level inference experiments for
`Qwen/Qwen3-0.6B`. The goal is to learn GPU engineering by measuring real
prefill/decode bottlenecks, writing targeted kernels, and retaining PyTorch or
NVIDIA libraries whenever they remain faster.

## What we built

- Vector add with coalescing, masks, block/warp sweeps, bandwidth measurement,
  and add+SiLU fusion.
- Standalone SwiGLU plus gate-projection + up-projection + SwiGLU fusion, with
  separate tiled-prefill and M=1 decode paths.
- Qwen-ordered RMSNorm and residual-add + RMSNorm with FP32 reduction and the
  model's cast-before-weight boundary.
- Q/K RoPE and experimental Q/K RMSNorm + RoPE + direct static K/V-cache write.
- Fused row softmax, tiled GEMM, shape-bucketed Qwen prefill GEMM, decode GEMV,
  and educational online tiled attention.
- Reproducible correctness, microbenchmark, model-ablation, natural-prompt,
  Nsight Systems, and generated-code inspection tools.
- Experimental W4A16 decode GEMV, packed INT4 LM-head/local top-k fusion, and
  an opt-in quantized-decode model adapter. GPU validation is pending; the A/B
  adapter retains dense weights, so whole-model memory savings are not claimed.

No specialized external kernel repository was copied or integrated. The project
does use PyTorch, Triton, Transformers, cuBLAS/CUTLASS dispatch, and PyTorch
SDPA/FlashAttention as dependencies and strong baselines.

## Measured RTX 3090 result

Batch 1, BF16, SDPA, context 2,048, five CUDA-event samples after warmup:

| Metric | PyTorch baseline | Best candidate | Improvement |
|---|---:|---:|---:|
| Prefill / TTFT approximation | 63.58 ms | 50.54 ms | 20.5% lower; 25.8% more throughput |
| Decode / ITL | 22.11 ms/token | 19.17 ms/token | 13.3% lower; 15.4% more throughput |
| Prefill kernel launches | 1,593 | 718 | 55% fewer |
| Decode launches/token | 1,624 | 749 | 54% fewer |

The best prefill candidate combines Qwen RMSNorm with fused gate/up/SwiGLU.
The best decode candidate uses RMSNorm alone; fused MLP currently slows M=1
decode. “TTFT” here is model prefill latency, excluding server/network/tokenizer
overhead.

## What worked—and what did not

**Worked:** RMSNorm provides the strongest end-to-end gain. Fused gate/up/SwiGLU
wins isolated prefill shapes by 7–39%. The experimental norm+RoPE+cache write is
promising in isolation, but is not yet integrated into the model cache path.

**Did not work:** every custom BF16 decode GEMV lost to PyTorch/cuBLAS (1.14–1.67x
slower). Our educational attention loses to optimized SDPA, so SDPA remains the
backend. Fewer launches alone did not guarantee faster decode.

The pre-quantization candidate suites passed 241/241 GPU tests. This is not full model
equivalence: RMSNorm still changes some logits, and the combined candidate
matched only 3/4 natural-language smoke generations. Larger quality evaluation
is required. Nsight Compute counters were provider-blocked, so achieved
occupancy and hardware-counter rooflines are not claimed.

## Next targets

1. Integrate residual-add + RMSNorm at real decoder boundaries.
2. Integrate Q/K norm + RoPE + preallocated cache and verify every cache slot.
3. Add CUDA Graph replay after cache addresses become static.
4. Build INT8/INT4 weight-only decode projections, especially the LM head.
5. Use shape/phase dispatch: fused MLP for winning prefill shapes, cuBLAS for
   decode GEMV, and SDPA for attention.
6. Expand per-layer drift and quality evaluation before production claims.

## Start here

- [Latest results and limitations](results/profiling_20260901/REPORT.md)
- [Experiment roadmap](docs/EXPERIMENTS.md)
- [RunPod guide](docs/RUNPOD.md)
- [Prefill plan](docs/PREFILL_OPTIMIZATION_PLAN.md)
- [Fused projection + SwiGLU](learning/02_swiglu/FUSED_PROJECTION.md)
- [Qwen RMSNorm](learning/03_rmsnorm/QWEN_RMSNORM.md)
- [Decode kernels and static cache](learning/08_decode/README.md)
- [INT4 decode, LM head and compact top-k (unvalidated)](learning/09_int4/README.md)

On a prepared RunPod volume:

```bash
source /workspace/gpu-kernel-lab/scripts/activate_runpod.sh
python -m pytest learning/03_rmsnorm/test_qwen_rmsnorm.py \
  learning/02_swiglu/test_fused_projection_swiglu.py \
  learning/06_gemm/test_qwen_prefill_gemm.py \
  learning/08_decode/test_decode.py -q
```
