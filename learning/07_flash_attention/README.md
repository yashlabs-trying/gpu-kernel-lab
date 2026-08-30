# Lesson 7: Tiled Online Attention

Attention is:

```text
scores = Q @ Kᵀ / sqrt(D)
probabilities = softmax(scores)
output = probabilities @ V
```

A naive implementation materializes `[B,H,Sq,Sk]` scores and probabilities.
Their size grows quadratically with sequence length. This lesson implements an
educational FlashAttention-style forward pass that never stores either matrix.

## 1. Online softmax

K/V are visited in small `BLOCK_N` tiles. For each query row we retain only:

```text
m = maximum score seen so far
l = sum(exp(score - m)) for scores seen so far
acc = weighted V sum in the same scale
```

When a new score tile arrives:

```text
new_m = max(old_m, max(new_scores))
correction = exp(old_m - new_m)
new_l = old_l * correction + sum(exp(new_scores - new_m))
new_acc = old_acc * correction + exp(new_scores - new_m) @ V_tile
```

After all tiles:

```text
output = acc / l
```

The correction rescales old statistics whenever a larger maximum appears. This
makes tiled results mathematically equivalent to stable full-row softmax, subject
to floating-point order differences.

## 2. Kernel fusion and forward memory

One kernel performs:

```text
Q tile × K tileᵀ
→ scale
→ padding/causal mask
→ online softmax update
→ probability tile × V tile
→ final output
```

Scores and probability tiles live only on-chip. Global memory receives no full
score/probability matrix—only output and a compact `[B,H,Sq]` log-sum-exp tensor.
This changes intermediate storage from quadratic `O(Sq×Sk)` to linear saved
statistics plus fixed-size tiles.

## 3. Small tiles and backward recomputation

Training backward needs probability information, but saving all probabilities
would lose the memory advantage. FlashAttention saves compact forward state such
as output and log-sum-exp:

```text
LSE = running_max + log(running_sum)
```

During backward, score/probability tiles can be regenerated from Q and K using
the saved LSE, then immediately used to form gradients. This trades extra compute
for far less memory traffic/storage.

Our code saves LSE and implements the forward pass. A backward recomputation
kernel is **not implemented yet** and is not claimed. For inference, no backward
state is needed at all; an inference integration can omit LSE output.

## 4. Causal upper triangle and padding

For causal self-attention, query position `q` may only see keys `k <= q`:

```text
allowed: k <= q
masked:  k > q  (upper triangle)
```

The kernel replaces upper-triangular scores with `-inf`, so their exponential is
zero. Per-batch `query_lengths` and `key_lengths` also mask padded rows/columns.
This happens before both online reductions, preventing padding from changing max
or sum statistics.

Future optimization: causal tiles completely above the diagonal can be skipped
rather than loaded and masked. Diagonal tiles still need elementwise masks.

## 5. Asymmetric warp scheduling

Modern Hopper kernels can specialize warps into asymmetric roles:

```text
producer warps: move Q/K/V tiles (often using TMA)
consumer warps: execute matrix multiply and online softmax
```

This overlaps movement and computation without making every warp do identical
work. Our RTX 3090 is Ampere, not Hopper. It does not provide the same TMA and
warp-specialized execution path. The implemented Ampere kernel therefore tunes:

```text
BLOCK_M, BLOCK_N, num_warps, num_stages
```

`num_stages` allows pipelining; it is not evidence of Hopper warp specialization.
When testing Hopper later, we can create a separate architecture-specific kernel
and verify warp specialization/TMA in generated code and profiler traces.

## Hardware mapping and pressure

The grid is two-dimensional:

```text
axis 0 = query tile
axis 1 = batch × head
```

One program holds a Q tile, running max/sum vectors, and an FP32 output tile while
streaming K/V. Larger `BLOCK_M`, `BLOCK_N`, or head dimension increases registers
and shared memory. Too-small tiles add loop/launch overhead; too-large tiles lower
occupancy or spill. That is why benchmarking configurations is mandatory.

This educational kernel supports FP16/BF16 and head dimensions up to 256. It uses
the same Q/K head count; GQA head mapping and KV-cache paging are later integration
steps.

## Run on the Pod

```bash
source /workspace/gpu-kernel-lab/.venv/bin/activate
cd /workspace/gpu-kernel-lab/learning/07_flash_attention

pytest -q
python benchmark.py --dtype bfloat16 --output ../../results/attention_bf16_rtx3090.csv
python benchmark.py --dtype float16 --output ../../results/attention_fp16_rtx3090.csv
```

## Evidence checklist

- Does online output match the materialized reference?
- Are causal upper-triangle and variable-length padding correct?
- Are invalid positions excluded from max and sum?
- Are scores/probabilities absent from global output storage?
- Which tile/warp/stage configuration wins by sequence length?
- What are register count, shared-memory use, occupancy, and spills?
- Can fully masked causal tiles be skipped?
- On future Hopper hardware, does codegen prove TMA/warp specialization?
- For training, does a future recomputation backward match PyTorch gradients?

