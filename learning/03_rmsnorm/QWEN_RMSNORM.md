# Qwen-compatible fused RMSNorm: first implementation

Status: written, GPU validation pending. The previous Pod SSH endpoint returned
`container not found`. No new performance or correctness result is claimed.

## What changed?

The original teaching kernel rounds only after multiplying by the weight.
Qwen's reference rounds normalized values back to the input dtype FIRST:

```text
x -> FP32 -> square -> mean -> add epsilon -> rsqrt
                         x * rsqrt
                             |
                    cast back to BF16/FP16
                             |
                       multiply weight
                             |
                            store
```

In `qwen_rmsnorm.py`, the explicit cast to the input element type and back to
FP32 preserves that rounding boundary. Multiplication then computes the product
of those rounded values and the weight, before output storage rounds again.
`enable_fp_fusion=False` prevents multiply/add contraction across relevant
arithmetic boundaries. It does not force PyTorch's reduction tree or rsqrt.
Thus **compatible operation ordering is not proven bitwise equality**.

## Five optimization decisions

1. **Row mapping:** one Triton program handles one contiguous last-dimension row.
   Q/K head norms have width 128; layer norms have width 1024. Leading dimensions
   flatten into rows. This implementation rejects noncontiguous input rather
   than silently adding a copy kernel.
2. **Reduction:** neighboring logical lanes load neighboring columns. Each
   program reduces squares with `tl.sum`; Triton chooses intra-warp shuffles and
   any cross-warp exchange. We do not manually assign physical lanes. Padding
   lanes contribute zero, and division uses the real width, not padded width.
3. **FP32:** square, sum, mean, epsilon and normalization use FP32. The explicit
   pre-weight cast reproduces the model's low-precision rounding point.
4. **Resources:** the block is the next power of two. Tests cover 2/4/8 warps;
   default is 4 until measured. Width is capped at 4096 for this model-targeted
   implementation. Register count, spills and occupancy are not yet measured.
5. **Fusion:** one input load, one weight load and one output store per element;
   no full intermediate tensors. RoPE/residual/cache remain unchanged to isolate
   this experiment. No custom backward is supplied: inference only.

## Validation gates

The parameterized GPU suite has 180 standalone reference cases (3 dtypes,
5 shapes, 4 input scales, 3 warp counts), plus residual-fusion, layout, and
empty-row checks. It compares the
eager ordering exactly against the installed Transformers Qwen3RMSNorm, then
checks the Triton output with declared dtype tolerances. Passing tolerance is
not evidence of bitwise equality or full-model quality.

Run on the prepared GPU environment from the repository root:

```bash
python -m pytest learning/03_rmsnorm/test_qwen_rmsnorm.py -q
python profiling/model_workload.py --variant qwen_rms --output results/qwen_rms_candidate.json
python profiling/quality_probe.py --variant qwen_rms --output results/qwen_rms_generation.json
```

`qwen_rms` replaces only RMSNorm modules, leaving SwiGLU and SDPA untouched.
It records synthetic logit error and greedy-token agreement before timing.
The four-prompt generation probe is a smoke check, not a sufficient quality gate.

Before accepting this change, still needed: GPU execution, expanded real-prompt
and per-layer drift checks, interleaved baseline/candidate timing, and an Nsight
launch-count check. Start with RMSNorm alone; do not attribute gains from the
previous combined RMSNorm/SwiGLU experiment to this untested kernel.
