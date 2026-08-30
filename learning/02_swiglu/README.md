# Lesson 2: Fused SwiGLU

SwiGLU is used in the feed-forward/MLP part of many transformer models,
including the Qwen model in this project. This lesson starts after the two linear
projections have produced tensors called `gate` and `up`.

## 1. The mathematics

For each element:

```text
sigmoid(x) = 1 / (1 + exp(-x))
SiLU(x)    = x * sigmoid(x)
output     = SiLU(gate) * up
```

So our complete elementwise expression is:

```text
output[i] = gate[i] * sigmoid(gate[i]) * up[i]
```

Example (rounded):

```text
gate = 1, up = 2
sigmoid(1) = 0.7311
SiLU(1) = 0.7311
output = 0.7311 * 2 = 1.4622
```

In a transformer MLP, `gate` and `up` normally come from different projections:

```text
gate = input @ gate_weight
up   = input @ up_weight
hidden = SiLU(gate) * up
output = hidden @ down_weight
```

This experiment fuses only `SiLU(gate) * up`. It does not yet fuse either matrix
multiplication. Keeping that boundary clear prevents us from claiming more than
we implemented.

## 2. Coalesced memory access

As in vector addition, each program owns one consecutive range:

```python
offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Neighboring lanes load neighboring `gate` elements, load neighboring `up`
elements, and store neighboring outputs. All three streams are coalesced for
contiguous tensors.

The wrapper accepts tensors of any shape because the operation is elementwise,
but their storage must be contiguous. `numel()` safely flattens the contiguous
memory conceptually; output retains the original shape.

## 3. Grid, block size, and boundary mask

```text
grid = ceil(number of elements / BLOCK_SIZE)
mask = offsets < number of elements
```

The mask prevents the final program from touching invalid addresses. Tests use
sizes `1, 7, 127, 128, 129, 1,000, 1,003, and 100,003` so boundary behavior is
actually exercised.

The benchmark crosses:

```text
BLOCK_SIZE: 128, 256, 512, 1024
num_warps:  2, 4, 8
```

That gives 12 Triton configurations per shape. Block size controls elements per
program; `num_warps` controls CUDA warps cooperating on that program. More of
either is not automatically faster because resource use, occupancy, scheduling,
and tensor size interact.

## 4. Accuracy and dtype

Sigmoid is nonlinear and involves an exponential, so SwiGLU does not generally
produce bit-identical results across implementations. Our kernel explicitly
loads stored values and converts them to FP32:

```python
gate = tl.load(...).to(tl.float32)
up = tl.load(...).to(tl.float32)
```

It calculates sigmoid and both multiplies in FP32, then the store rounds the
answer to the output tensor's dtype. This is important for BF16 and FP16, which
have fewer representable values.

Correctness is checked against PyTorch with dtype-appropriate tolerances:

```text
FP32: rtol 1e-5, atol 1e-6
FP16: rtol 2e-3, atol 2e-3
BF16: rtol 2e-2, atol 2e-2
```

Tolerance is not permission for arbitrary error. On the GPU we will also inspect
maximum/typical differences if a test fails and adjust only with evidence.

## 5. Math implementation

The kernel keeps the formula visible:

```python
sigmoid_gate = tl.sigmoid(gate)
output = (gate * sigmoid_gate) * up
```

This uses Triton's sigmoid implementation. Later experiments can study exact
versus approximate exponential math, but first we preserve a clear reference and
measure its error. A faster approximation is useless if model accuracy degrades.

## 6. Fusion opportunity and memory traffic

An eager two-operation implementation conceptually does:

```text
temporary = SiLU(gate)  # kernel 1
output = temporary * up # kernel 2
```

It has two launches and approximately five element transfers:

```text
read gate + write temporary + read temporary + read up + write output
```

Our fused kernel performs one launch and three element transfers:

```text
read gate + read up + write output
```

It eliminates the temporary tensor's write and read. For BF16, ideal useful
traffic is `3 * N * 2` bytes. For FP32 it is `3 * N * 4` bytes.

The benchmark reports **useful fused bandwidth**:

```text
useful GB/s = (3 * N * bytes per element) / elapsed seconds / 1e9
```

This gives both implementations the same useful-work numerator, allowing their
latencies and useful throughput to be compared. It does not claim exact physical
DRAM traffic—caches and framework/compiler behavior must be profiled separately.

An even larger future fusion could combine the gate/up projections with SwiGLU,
or combine SwiGLU with the down projection. That requires matrix-multiplication
tiling and has very different complexity, so it is intentionally not hidden in
this first activation kernel.

## 7. Run on the GPU Pod

From the repository root after pulling the commit:

```bash
source /workspace/gpu-kernel-lab/.venv/bin/activate
cd /workspace/gpu-kernel-lab/learning/02_swiglu

pytest -q
python benchmark.py --dtype bfloat16 --output ../../results/swiglu_bf16_rtx3090.csv
python benchmark.py --dtype float32 --output ../../results/swiglu_fp32_rtx3090.csv
```

We accept performance results only after all accuracy tests pass.

## 8. Review checklist

- Are all three memory streams coalesced?
- Does the mask protect every load and store?
- Were all block/warp pairs benchmarked rather than guessed?
- Does each dtype stay within its tested accuracy tolerance?
- Is FP32 used for nonlinear intermediate math?
- Is the eager reference mathematically identical?
- Did fusion remove the intermediate tensor and one kernel launch?
- Are latency and useful GB/s both reported?

Only after this review and a real GPU run should we move to GELU.

