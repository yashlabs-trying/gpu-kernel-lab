# Lesson 5: Fused Row-Wise Softmax

Stable softmax for a row `x` is:

```text
m = max(x)
e = exp(x - m)
output = e / sum(e)
```

Subtracting the maximum prevents large logits from overflowing `exp`.

## The five optimization problems

1. **Efficient reductions.** `tl.max` and `tl.sum` are tree reductions across
   lanes/warps. Triton selects warp exchange and cross-warp communication. We
   tune warps because both reductions depend on that mapping.
2. **Load once.** One program loads its entire row once. `x`, shifted values, and
   exponentials stay on-chip through max, exp, sum, and divide; only the final
   probabilities are stored. Masked columns load `-inf`, whose exponential is
   zero, so they cannot affect either reduction.
3. **FP32.** BF16/FP16 input is converted to FP32 before max, exponential,
   summation, and division. Only the final store rounds to the input dtype.
4. **Block/register pressure.** The minimum block is
   `next_power_of_two(columns)`: 3,000 columns require 4,096 lanes. Because `x`
   and `exp(x-max)` remain live, wide rows use many registers and may spill or
   reduce occupancy. We test the minimum block, the next larger block, and
   `2/4/8` warps. Above the teaching cap of 65,536, use tiled/multi-stage logic.
5. **Fusion.** Standalone softmax still reads and writes an attention matrix.
   FlashAttention instead forms score tiles from Q/K, maintains online softmax
   statistics, combines V, and writes only output—without materializing full
   score or probability matrices. That is our endpoint, not something this
   standalone kernel claims to implement.

Conceptually, the memory path is:

```text
LOAD ROW ONCE -> MAX -> EXP -> SUM -> DIVIDE -> STORE ONCE
```

Tests cover irregular widths, FP32/FP16/BF16, sum-to-one, and extreme logits.

Run:

```bash
cd /workspace/gpu-kernel-lab/learning/05_softmax
pytest -q
python benchmark.py --output ../../results/softmax_bf16_rtx3090.csv
```

