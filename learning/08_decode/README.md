# Decode candidates: exact-shape GEMV and static KV-cache fusion

Status: implemented, GPU validation pending. No speedup is claimed.

## GEMV, not a padded GEMM

Decode projection has M=1. `decode_gemv.py` consumes an exact `[K]` vector and
row-major `[K,N]` matrix. Programs own disjoint output columns and reduce K in
FP32, so the normal path is one launch with no atomics. Masking applies only at
the N/K tile boundary; no padded tensor is allocated and no padded output exists.

The optional Split-K path writes disjoint FP32 partials and performs one second
reduction launch. This is safer and deterministic relative to FP32 atomics, but
it adds workspace traffic and a launch. It is a candidate only for deep-K shapes
and only if it beats both the one-launch kernel and `torch.mv`/cuBLAS.

We do not claim register-pressure control until compiler metadata is inspected.
Keeping a BN-vector accumulator live is the relevant tradeoff: larger BN gives
fewer blocks and more per-program state; smaller BN rereads X across more blocks.

## Q/K norm + RoPE + cache write

`fused_qkv_cache.py` handles Qwen's decode tensors directly:

```text
Q [B,16,128] -> RMSNorm -> RoPE -> Q output
K [B, 8,128] -> RMSNorm -> RoPE -> K cache[B,8,position,128]
V [B, 8,128] ------------------> V cache[B,8,position,128]
```

Each program owns one head. It performs the width-128 FP32 reduction, preserves
Qwen's pre-weight dtype rounding, indexes compact RoPE tables directly by the
cache position, applies the half-rotation, and writes exact
logical elements. The K program also copies its matching V row. Writes across D
are contiguous/coalesced. The cache is preallocated, so no old cache is copied.

The cache **cannot live permanently in shared memory**. Shared memory belongs to
one thread block for one kernel's lifetime and is tens of KiB, whereas the model
cache is hundreds of MiB. A later attention kernel can stage the currently used
K/V tile through L2/shared memory/registers. Persistent cache storage remains
global VRAM (or a paged global layout).

## Validation and acceptance

```bash
python -m pytest learning/08_decode/test_decode.py -q
python profiling/decode_candidates.py --output results/decode_candidates.json
```

Acceptance requires exact cache-slot checks, unchanged untouched slots, declared
numerical tolerances, multiple positions/batches, PTX/resource inspection, and
clean model decode timing. The fused path is not installed into Transformers yet.
First it must pass on a live GPU and then be adapted to the exact cache API/layout.
