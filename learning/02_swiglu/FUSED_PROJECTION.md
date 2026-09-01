# Gate + Up projections + SwiGLU fusion

Status: implemented and wired as the process-local `fused_mlp` model variant;
GPU validation pending. No performance claim is made.

## Removed global-memory path

Eager execution conceptually does:

```text
X @ Wgate -> write/read gate
X @ Wup   -> write/read up
SiLU      -> write/read activation
multiply  -> write hidden
```

The candidate does:

```text
load X tile once
  |-> dot with Wgate -> FP32 gate accumulator --|
  `-> dot with Wup   -> FP32 up accumulator   --+-> ordered casts -> SwiGLU
                                                       |
                                              one global output store
```

Weights must still be read: gate and up are different matrices. Reusing X and
eliminating three full intermediate streams is the achievable traffic reduction.

## Register and shared-memory strategy

Two accumulator tiles are simultaneously live, so copying a normal GEMM's large
tile is dangerous. The autotune search deliberately limits tiles to BM 16--64
and BN 32--64. This trades additional programs/X reuse against accumulator
register pressure. The decode M=1 path uses BN 16--64.

Triton decides how dot operands are staged through registers/shared memory and
which vector load instructions are generated. Source code cannot force values
to stay in the “closest” level indefinitely, and manually spilling accumulators
to shared memory may be slower. We accept a configuration only after inspecting:

- registers/thread and static shared memory;
- compiler spill metadata and runtime local-memory traffic;
- achieved occupancy when provider counters permit;
- exact model-shape latency against eager, `torch.compile`, and cuBLAS dispatch.

## Numerical order

Qwen eager produces stored low-precision gate/up projections, stored SiLU, then
the low-precision multiplication. The default `model_ordered=True` inserts those
dtype conversions inside the kernel without writing global intermediates:

```text
FP32 dot -> input dtype -> FP32 SiLU -> input dtype -> multiply -> output dtype
```

This preserves operation boundaries but does not promise bitwise equality,
because GEMM accumulation order and sigmoid implementations can differ.
`model_ordered=False` is an experimental higher-precision epilogue and must not
be substituted silently.

## Exact shapes and padding

Logical M/N/K are never padded tensors. Boundary masks protect irregular shapes.
Execution tiles remain aligned because Tensor Cores require tile structure;
“no padding” must not mean scalarizing the hardware. M buckets reduce autotune
and binary variants, while stores remain limited to exact logical rows/columns.

## Run the acceptance gate

```bash
python -m pytest learning/02_swiglu/test_fused_projection_swiglu.py -q
python profiling/fused_mlp_benchmark.py --output results/fused_mlp.json
python profiling/model_workload.py --variant fused_mlp \
  --output results/fused_mlp_model.json
python profiling/quality_probe.py --variant fused_mlp \
  --output results/fused_mlp_generation.json
```

Only integrate if correctness passes and end-to-end prefill/decode improve.
The earlier standalone SwiGLU microbenchmark is not sufficient evidence.
