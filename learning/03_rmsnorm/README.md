# Lesson 3: RMSNorm and Fused Residual RMSNorm

RMSNorm is our first **reduction kernel**. Vector add and SwiGLU could calculate
each output independently. RMSNorm cannot produce any normalized element until
it has combined information from every element in the same row.

## 1. Mathematics

For one row `x` with hidden size `H` and learned weight `w`:

```text
mean_square = (x[0]^2 + x[1]^2 + ... + x[H-1]^2) / H
inverse_rms = 1 / sqrt(mean_square + epsilon)
output[i] = x[i] * inverse_rms * w[i]
```

Unlike LayerNorm, RMSNorm does not subtract the row's mean. It uses the root mean
square as its scale.

For an input shaped `(batch, sequence, hidden)`, every `(batch, sequence)` pair
is one logical row. Contiguous storage lets us flatten the leading dimensions:

```text
rows = batch * sequence
columns per row = hidden
```

## 2. Optimization 1 — Row mapping

Our launch grid is:

```text
grid = (number_of_rows,)
```

Therefore:

```text
program 0 -> complete row 0
program 1 -> complete row 1
program 2 -> complete row 2
...
```

Within a program, logical lanes own columns:

```python
columns = tl.arange(0, BLOCK_SIZE)
row_start = row * hidden_size
addresses = row_start + columns
```

This is coalesced because neighboring lanes access neighboring columns. When
`hidden_size < BLOCK_SIZE`, the program does not execute a Python loop. It creates
a power-of-two group of lanes and masks columns outside the row:

```python
mask = columns < hidden_size
```

Masked loads use `other=0.0`, so nonexistent columns add zero to the reduction.
Masked stores write nothing.

One program per row is a good simple mapping when the complete row fits within a
manageable block and register budget. It also avoids communication between
different programs, which Triton cannot perform inside one ordinary kernel
launch.

## 3. Optimization 2 — Reduction efficiency and communication

The heart of the kernel is:

```python
squared_sum = tl.sum(x * x, axis=0)
```

Conceptually, lanes begin with separate squared values and combine them in a
tree:

```text
8 values -> 4 partial sums -> 2 partial sums -> 1 row sum
```

We do not manually tell thread 3 to talk to thread 4. Triton's compiler lowers
`tl.sum` to an efficient GPU reduction. Values within a warp can be exchanged
with warp-level instructions; partial results across warps may require shared
communication selected by the compiler. `num_warps` changes this organization,
which is why it must be measured.

The single resulting scalar `reciprocal_rms` is then used by all logical lanes in
that program to normalize their elements.

## 4. Optimization 3 — FP32 accumulation

Squaring and summing hundreds or thousands of low-precision numbers can magnify
rounding error or overflow. Inputs may be BF16 or FP16 in model inference, but we
immediately convert loaded values:

```python
x = tl.load(...).to(tl.float32)
squared_sum = tl.sum(x * x, axis=0)
```

Mean, epsilon addition, reciprocal square root, weight multiplication, and the
intermediate normalized value are FP32. The final store rounds back to the output
tensor's dtype. Tests compare FP32, FP16, and BF16 against a PyTorch reference
that also calculates the reduction in FP32.

## 5. Optimization 4 — Block size, warps, and register pressure

The smallest legal one-program block is:

```text
BLOCK_SIZE = next_power_of_two(hidden_size)
```

Examples:

```text
hidden  1,000 -> block 1,024 -> 24 masked lanes
hidden  1,024 -> block 1,024 -> no masked lanes
hidden  4,096 -> block 4,096 -> no masked lanes
```

We also test the next larger block to demonstrate whether extra masked lanes help
or hurt. Usually it adds no useful work and can increase resource cost.

The program keeps `x`, `x*x`, weights, and intermediate output values live around
the reduction. The compiler maps these values into registers and may spill when
pressure becomes too high. Larger hidden sizes and block sizes can therefore:

- reduce the number of resident programs/warps (occupancy);
- cause register spilling into much slower local memory;
- make cross-warp reduction more expensive.

The benchmark crosses each legal block with `num_warps = 2, 4, 8`. More warps
can accelerate a large reduction, but can also increase communication and
resource use.

This teaching kernel caps `BLOCK_SIZE` at 65,536. That cap is not a promise that
every size below it performs well. Very large rows should use a multi-stage
design: several programs calculate partial sums, another stage combines them,
and a final stage normalizes. Profiling register spills and occupancy tells us
when to switch; guessing does not.

## 6. Optimization 5 — Residual fusion

Transformer blocks commonly need:

```text
combined = x + residual
normalized = RMSNorm(combined)
```

Separate kernels conceptually move:

```text
residual add: read x + read residual + write combined
RMSNorm:     read combined + read weight + write normalized
```

Our fused kernel loads `x` and `residual`, adds them in FP32 registers, reduces
the unrounded combined values, and normalizes without rereading `combined` for
the reduction. Only the saved residual output is rounded back to its storage
dtype. The PyTorch fused reference follows this same accuracy rule.
It emits two outputs because the updated residual stream is needed later:

```text
residual_output = x + residual
normalized_output = RMSNorm(residual_output)
```

This removes a separate launch and avoids one trip through an intermediate input
path. Fusion increases live values and therefore register pressure, so its real
benefit must be measured—especially for large hidden sizes.

## 7. What each file proves

- `rmsnorm.py`: clear PyTorch references plus standalone and fused Triton kernels.
- `test_rmsnorm.py`: irregular masks, several hidden sizes, three dtypes, and both
  fused outputs compared with PyTorch.
- `benchmark.py`: row/hidden shapes, minimum and oversized blocks, `2/4/8` warps,
  and eager versus fused CUDA-event latency.

Tests include hidden sizes `7`, `127`, `128`, `1,000`, `1,024`, and `4,096`.
The 1,000-wide rows prove the mask works; 1,024 and 4,096 represent model-like
power-of-two hidden dimensions.

## 8. Run on the GPU Pod

```bash
source /workspace/gpu-kernel-lab/.venv/bin/activate
cd /workspace/gpu-kernel-lab/learning/03_rmsnorm

pytest -q
python benchmark.py --dtype bfloat16 --output ../../results/rmsnorm_bf16_rtx3090.csv
python benchmark.py --dtype float32 --output ../../results/rmsnorm_fp32_rtx3090.csv
```

Custom shapes use `ROWSxHIDDEN`:

```bash
python benchmark.py --shapes 1x1024 128x1024 2048x1024
```

## 9. Review checklist

- Is exactly one program mapped to each row?
- Are column loads/stores coalesced and masked?
- Do invalid lanes contribute zero to `tl.sum`?
- Is the square-and-sum accumulation FP32?
- Does `BLOCK_SIZE` cover the row without being chosen blindly?
- Are `2`, `4`, and `8` warps measured?
- Do larger blocks reveal latency or register-pressure regressions?
- Does fused residual RMSNorm match both expected outputs?
- Does fusion improve real latency rather than only reducing operations on paper?

Only GPU correctness and profiler evidence can complete the experiment.
