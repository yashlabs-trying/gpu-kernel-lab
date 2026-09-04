# Integrated static KV cache for Qwen3-0.6B

This stage turns the earlier isolated Q/K-normalization + RoPE + cache-write
kernel into an actual model attention path. It is an opt-in inference experiment,
not a general Transformers cache replacement.

## What changed

The two persistent allocations are physically:

```text
K[layer, batch, kv_head, position, head_dimension]
V[layer, batch, kv_head, position, head_dimension]
```

For Qwen3-0.6B that is `[28, 1, 8, capacity, 128]` for each allocation. Every
Transformers static-cache layer points to its slice of those buffers. No layer
rebinds its K/V pointer during decode.

Prefill uses the original Transformers path to populate these buffers. For one
decode token, the installed attention path does:

```text
Q/K/V projections
        |
        +-- Q RMSNorm -- RoPE --> Q attention input
        |
        +-- K RMSNorm -- RoPE --> K[layer, batch, head, position, :]
        |
        +-- V -----------------> V[layer, batch, head, position, :]
        |
        +-- open position in persistent additive attention mask
```

The normalization reductions accumulate FP32. The Qwen dtype boundary before
weight multiplication is retained. RoPE tables are computed once at installation
and stored compactly as `[capacity, D/2]`; the model's decode-time rotary module
returns persistent dummy tensors because the fused attention consumes the
precomputed tables directly. Prefill still uses the original rotary module.

The mask has fixed `[B,1,1,capacity]` shape and pointer. It starts with zero for
the prefix and the lowest finite BF16 value for the unused suffix. The fused
layer writes zero at the current position in the same launch as K/V. Later
layers repeat that idempotent write, but launch no new mask kernel. Unlike the
old benchmark loop, there is no `torch.cat` that grows an attention mask.

A one-element CUDA tensor stores the current position. Its pointer is fixed.
One tiny post-step kernel increments it. This avoids allocating `arange` and new
position tensors on every step. It does not yet eliminate the position-update
launch; a future graph can capture it.

## What is removed in decode

- No growing K or V concatenation.
- No copying the old K/V history to a new allocation.
- No materialized rotated-K tensor between RoPE and cache update.
- No growing attention-mask tensor.
- No per-step RoPE table generation.
- No `index_copy_` K/V update and no per-layer cache-length increment.

Raw Q, K and V projection outputs still exist. Q is required by attention; raw
K/V feed the fused transform/write. Attention output and logits are also still
ordinary model outputs. Static KV addresses are necessary for CUDA Graphs, but
the entire decode is **not graph-captured yet**. Token/logit workspaces and all
other graph inputs must also be made graph-stable before claiming that.

## Why static cache alone can be slower

Transformers `StaticCache` fixes addresses but still performs separate Q/K norms,
RoPE, `index_copy_` K/V writes and cache-length updates. A fixed-capacity
attention mask can also choose a different attention kernel or make attention
process masked capacity. Preallocation is an enabling mechanism, not itself a
latency guarantee. This is why the benchmark has three variants:

1. original `DynamicCache` with growing mask;
2. Transformers static cache with fixed mask/position;
3. static cache plus the fused direct-write attention integration.

Each context is run in a separate process so capacity equals context plus the
16 measured decode steps. Results from a maximum-capacity server reserve can
differ, particularly at short prefixes.

## Correctness boundary

The integration currently requires:

- Qwen3-0.6B geometry: B=1, Hq=16, Hkv=8, D=128;
- CUDA BF16, eval mode and inference-only execution;
- exactly one decode token per call;
- eager, sequential calls with explicit fixed `position_ids` and additive mask;
- prefill before decode; no sliding-window layers;
- capacity known before allocation and no overflow.

Unsupported calls fall back during prefill, but the decode runner deliberately
does not support `.generate()`, beam search, batch changes, concurrent forwards,
cache cropping/reordering, sliding windows, or speculative decoding. Static-cache
length metadata remains at the prefill length because the fused path bypasses
`StaticLayer.update`; callers must use the persistent position tensor. The
quality probe compares dynamic, static-only and fused execution on identical
teacher-forced tokens and asserts every K/V/mask/position address remains fixed.

Changing the attention shape and fusing low-precision arithmetic can alter
rounding even when the algorithm is logically equivalent. Argmax agreement and
loss must therefore be measured; passing kernel-level numerical tolerance is
not sufficient.

## Reproduction

After sourcing `scripts/activate_runpod.sh`:

```bash
python -m pytest -q learning/08_decode/test_decode.py learning/11_static_kv/test_static_kv.py
python profiling/static_kv_quality.py --output results/static_kv_quality.json
python profiling/static_kv_benchmark.py --lengths 2048 --output results/static_kv_2048.json
python profiling/static_kv_benchmark.py --hybrid --lengths 2048 --output results/static_kv_hybrid_2048.json
```

Use one context per benchmark invocation if you want capacity equal to that
context plus decode steps. Packing, cache allocation and prefill are outside the
decode event timing, but mask updates, position advance, attention and argmax
are inside. Report inverse-latency token rate as serial throughput only.
