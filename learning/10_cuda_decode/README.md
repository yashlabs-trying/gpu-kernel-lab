# Decode stage 2: warp reductions, INT8 and combined projections

These are **opt-in experiments**, not default inference replacements. Start with
the result report before choosing a kernel. Passing arithmetic tests does not
establish model quality or end-to-end speed.

## 1. The row and the warp

For one token, a linear layer computes `y[N] = W[N,K] @ x[K]`. Weight storage is
row-major: neighboring K elements are neighbors in memory.

One warp owns one output row. Lane 0 reads K indices 0, 32, 64, ...; lane 1
reads 1, 33, 65, ...; and so on. At each iteration, adjacent lanes read adjacent
weights. Each lane keeps a partial dot product in FP32. Five
`__shfl_down_sync` steps (16, 8, 4, 2, 1) reduce those partial sums. Lane 0
writes the output. No shared-memory reduction or `__syncthreads()` is needed.

All 32 lanes execute each shuffle, even if a lane had no valid K elements.
That makes the full participation mask correct. The `row < outputs` condition
is uniform across a warp. Output-row tails are skipped; no padded weight rows
are stored or processed. The BF16 path also supports non-multiple-of-32 K.
The INT8 format currently requires K divisible by its group size (128 here).

One warp **per row** does not require one warp **per block**. We compare blocks
with 1, 4 and 8 independent warps. More independent warps can hide memory latency;
one-warp blocks can instead hit the hardware's block-count limit. There is no
universal winner. This is warp-centric reduction, not producer/consumer warp
specialization or a Hopper/Blackwell asynchronous pipeline.

The persistent candidate caps the grid at twice the SM count and uses a
grid-stride loop over rows. It can reduce scheduling overhead or reduce useful
parallelism; measure it. The kernel currently uses scalar coalesced accesses,
not explicit `float4`/`int4` vector loads. It does not use native INT8 Tensor Core
arithmetic or `dp4a`; weights are dequantized and multiplied in floating point.

## 2. Activation-aware INT8, precisely

`activation_int8.py` implements **AWQ-inspired W8A16 channel-scale search**.
It is not the complete AWQ algorithm, an AWQ checkpoint format, SmoothQuant,
or W8A8 activation quantization.

For each input channel, compute mean absolute activation on calibration tokens.
Try channel scales `s = mean_abs(x)**alpha`, with alpha in `{0, 0.5, 1}`:

```
scaled weight = W * s
INT8 weights  = round(scaled weight / group_scale), limited to [-127,127]
runtime input = x / s
output        = dequantized_scaled_weight @ runtime_input
```

Choose the scale giving the lowest FP32 local projection-output MSE on calibration
tokens. Alpha 0 includes ordinary round-to-nearest quantization as a candidate;
activation scaling is not assumed to win. There is no clipping search or global
model-level optimization. Four calibration passages and at most 32 rows per
projection are only a small initial experiment. The quality passages are separate.

Storage is 1 byte/weight plus one FP32 scale per 128 weights and an FP32 inverse
channel scale per input column. The inverse scale is applied inside the kernel;
there is no separate scaled-activation allocation or launch. Calibration and
packing happen before benchmarking.

The A/B adapter retains original BF16 weights for prefill. Therefore **resident
model memory increases**, even though the quantized weight representation is
smaller. The LM head remains BF16 in `qwen_rms_int8_cuda`; the optional
`qwen_rms_int8_cuda_int4_head` variant installs the previous INT4 head separately.

## 3. What is actually combined?

- **QKV:** concatenate Q/K/V weight rows offline. One CUDA launch produces their
  concatenated outputs; Q/K/V are views, not copied intermediate tensors.
  The existing Q/K normalization, RoPE, and cache update still run separately.
  Different output warps can reread x; combining launches does not imply x is
  loaded only once across the entire GPU.
- **Gate + up + SwiGLU:** one warp owns one gate/up output pair, reuses each x
  value in both dot products, then applies SiLU(gate)*up and writes only the
  final activation. Gate and up are rounded to BF16 before activation; SiLU's
  result is rounded to BF16 before multiplication, matching eager dtype
  boundaries. FP32 reduction order still differs from cuBLAS.
- **Bias:** this Qwen configuration has bias-free projections, so there is no
  bias addition to fuse. Biased models are explicitly rejected by the adapter.
- **Vocabulary:** `qwen_rms_int4_head` isolates the existing quantized head while
  retaining RMSNorm fusion and all other BF16 projections. Compact top-k remains
  a separate kernel experiment, not part of Transformers generation.

The combined QKV adapter relies on the inspected Transformers Qwen3 call order
Q then K then V, and resets its temporary output views on every attention call.
It is restricted to eager, batch-one, single-token decode, with no concurrent
forwards. It is not a CUDA-Graph-safe serving implementation. Prefill uses the
original projections. Other Qwen/Transformers implementations need validation.

After microbenchmarking, `qwen_rms_cuda_hybrid` isolates the promising pieces:
BF16 combined QKV (4 warps/block), INT8 gate/up/SwiGLU (8 warps/block), and
unchanged PyTorch output/down projections and LM head. It avoids the slower
INT8 output/down kernels. This is a measured-candidate combination, not a
guarantee that isolated improvements will compose into a model-level win.

## 4. Split-KV attention

`split_kv.py` is a separate Triton prototype. Each program handles one Q head
and a chunk of valid KV positions. For each chunk it computes stable softmax,
stores a normalized partial output and log-sum-exp, then a second kernel merges:

`output = sum(exp(local_lse - max_lse) * local_output) / sum(exp(local_lse - max_lse))`.

Simply summing partial attention outputs would be incorrect. Splitting positions
is intra-GPU sequence partitioning, not multi-GPU tensor parallelism. Global KV
storage remains in VRAM; only chunks live on chip. One warp per entire attention
head is not mandated: the 2D score/value reductions use Triton's mapping.

The prototype supports Q `[Hq,D]`, cache `[Hkv,S,D]`, GQA, BF16/FP16, D=64/128,
unit final cache stride, and non-power-of-two S. **Every supplied position must
be valid past/current context.** It has no padding mask, window mask, dropout,
or future positions. A single final-position query uses `is_causal=False` in
the SDPA reference because it should see the whole supplied cache. The merge
is limited to 256 chunks. This kernel is not installed into the model.

## 5. Tests and measurements

Run after activating the RunPod environment:

```bash
python -m pytest -q learning/10_cuda_decode/test_cuda_decode.py learning/10_cuda_decode/test_split_kv.py
python profiling/decode_stage2_benchmark.py --output results/stage2_micro.json
python profiling/model_workload.py --variant qwen_rms_int4_head --lengths 2048 4096 --output results/head_latency.json
python profiling/model_workload.py --variant qwen_rms_int8_cuda --lengths 2048 4096 --output results/int8_latency.json
python profiling/decode_stage2_quality.py --variant qwen_rms_int8_cuda --output results/int8_quality.json
```

Compilation requires CUDA/NVCC and Ninja. Set `TORCH_EXTENSIONS_DIR` under
`/workspace/.cache`; the activation helper now does so. Compilation is outside
timing. Rebuild for a different GPU architecture. Report model loss/argmax drift
separately from kernel error against dequantized weights. Compare whole-model
latency with RMSNorm already enabled, not only with an older unoptimized model.

References: [NVIDIA warp primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/),
[AWQ](https://github.com/mit-han-lab/llm-awq),
[Flash-Decoding](https://pytorch.org/blog/flash-decoding/).
