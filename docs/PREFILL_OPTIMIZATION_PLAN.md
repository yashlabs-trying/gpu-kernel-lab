# Prefill optimization implementation order

## Implemented candidates (GPU validation pending)

### 1. Qwen RMSNorm

`learning/03_rmsnorm/qwen_rmsnorm.py` now contains:

- standalone Qwen-compatible RMSNorm;
- residual addition + following RMSNorm in one kernel;
- FP32 square/reduction/normalization;
- an explicit low-precision residual write and Qwen pre-weight rounding point;
- one program per contiguous row and masked power-of-two padding.

The residual fusion replaces this real decoder boundary:

```text
sublayer output + residual -> write residual stream -> read it -> RMSNorm
```

with one load of each input and one kernel. It returns both the residual stream
and normalized tensor because both are required by the next decoder stage.
GELU/SiLU is deliberately not appended: it is not adjacent to Qwen's RMSNorm.
RoPE is the next meaningful fusion for the width-128 Q/K norms, after this gate.

### 2. Shape-bucketed prefill GEMM

`learning/06_gemm/qwen_prefill_gemm.py` targets the checkpoint's projection
shapes. M is bucketed to 16/32/.../4096 for autotune/cache reuse, while masks
avoid allocating or multiplying a physically padded input. The config search
covers tile shapes, warps, stages and grouped-M scheduling. Accumulators remain
FP32 and outputs are written once.

This is for prefill M>=13. Decode M=1 needs a GEMV design and is postponed.
Split-K is also postponed: Qwen's K values (1024/3072) are not automatically
deep enough to justify a second reduction/atomics. It will be added only if an
exact-shape profile shows under-parallelization.

## Acceptance gate

```bash
python -m pytest learning/03_rmsnorm/test_qwen_rmsnorm.py \
  learning/06_gemm/test_qwen_prefill_gemm.py -q
python profiling/prefill_candidates.py \
  --output results/prefill_candidates.json
python profiling/model_workload.py --variant qwen_rms \
  --output results/qwen_rms_model.json
python profiling/quality_probe.py --variant qwen_rms \
  --output results/qwen_rms_generation.json
```

Acceptance requires tests, layer/logit/generation checks, and repeated model
timings. GEMM must beat `torch.mm` at an exact shape before any projection is
replaced. Autotuning time is never included in benchmark samples. Reported
Triton timings are hot-kernel measurements and do not replace end-to-end timing.

## Still pending

1. Execute on a live GPU; the old endpoint currently says `container not found`.
2. Inspect compiled PTX/SASS and Nsight launch metadata. Do not claim `float4`,
   warp-shuffle strategy, register reuse, spills, occupancy, or asynchronous
   copies before inspection/counter evidence.
3. Add a process-local decoder-layer ablation for residual+norm only after exact
   residual-stream tests pass.
4. Fuse model-exact Q/K norm + RoPE, then profile launch/time removal.
5. Keep PyTorch SDPA and untargeted GEMMs as the baseline.
