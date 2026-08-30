# Lesson 4: RoPE for Q and K

RoPE rotates pairs of query/key features according to token position. Qwen's
half-rotation convention pairs dimension `i` with `i + D/2`:

```text
out_first  = first*cos - second*sin
out_second = second*cos + first*sin
```

## The five optimization questions

1. **Layout.** Q and K are contiguous `[B,H,S,D]`; `D` changes fastest in
   memory. One program handles one `(batch, head, position)` row. Q and K may
   have different `H` under grouped-query attention, but share `B,S,D`.
2. **Coalescing.** Lanes cover consecutive pair indices. They read a contiguous
   first half and contiguous second half, then store both contiguous halves.
3. **Sin/cos reuse.** Tables are compact `[S,D/2]`, not duplicated `[S,D]`.
   Each lane loads one cosine and sine and reuses them for both rotated outputs.
   Values are precomputed once by `build_rope_tables`, never recomputed in the
   kernel.
4. **Q + K together.** The grid contains all Q rows followed by all K rows. A
   single kernel launch selects the correct pointer per program. This removes a
   launch, but each row still needs its positional table values.
5. **Fusion endpoint.** This is not yet fused with Q/K projection, KV-cache
   writes, or attention. The eventual goal is to rotate Q immediately before
   attention and rotate K as it enters the cache, avoiding materialized rotated
   intermediates.

`PAIR_BLOCK = next_power_of_two(D/2)` and the mask handles non-power-of-two pair
counts. We sweep `num_warps=2,4,8`; larger is not automatically better. All math
uses FP32 intermediates and stores back to the model dtype.

Run:

```bash
cd /workspace/gpu-kernel-lab/learning/04_rope
pytest -q
python benchmark.py --output ../../results/rope_bf16_rtx3090.csv
```

Check layout and rotation convention before integrating: adjacent-pair RoPE and
half-split RoPE are not interchangeable even though both are called RoPE.

