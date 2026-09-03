# INT4 decode and vocabulary selection — experimental

Status: implemented, not GPU-validated. The previous Pod rejected SSH access.
No speedup or acceptable model quality is claimed.

## Formula and layout

For every row of a linear weight `[N,K]`, partition K into groups of 128:

```text
scale = max(abs(group)) / 7
q     = clamp(round(weight / scale), -7, 7)
weight_approx = q * scale
```

Zero groups use scale 1. Groups may also be 32 or 64. Two signed 4-bit values
are packed in each uint8 byte: even K in the low nibble, odd K in the high
nibble. Scales are FP32. K must divide evenly by the group size; there is no
physically padded K tensor. All current Qwen projection K dimensions satisfy
this constraint. Quantization is offline and chunked by rows to limit scratch
memory. This is round-to-nearest (RTN), not calibrated AWQ/GPTQ.

At group size 128, storage is `0.5 + 4/128 = 0.53125` bytes per weight versus
2 for BF16: **3.76x smaller, or 73.44% fewer bytes**, including scales.
For the 151936 x 1024 LM head this is 82,653,184 bytes versus 311,164,928 bytes.
These are buffer-size calculations, not measured whole-model memory savings.

## Fused decode computation

The kernel reads a packed byte once, unpacks two weights, applies their group's
scale and multiplies by BF16/FP16 input values. It accumulates in FP32 and writes
the final projection once. No full dequantized tensor is created on this path.

This is W4A16 weight-only computation, **not native INT4 Tensor Core arithmetic**.
Dequantization adds instructions; bandwidth savings do not guarantee a speedup.
The first candidate deliberately avoids Split-K, atomics, and extra partial
output buffers. Block width 8/16/32/64 is configurable for later resource tuning.

## LM-head fusion

`int4_lm_head_topk` fuses projection with local top-k per vocabulary tile. It
writes only candidate scores and token IDs, then merges candidates globally
using PyTorch reductions. Top-1 is argmax. Top-k supports k<=8.

Each global winner must be among its own tile's top-k, so the merge is exact
relative to the **quantized kernel's rounded logits**. It does not guarantee the
same tokens as the BF16 model. No full vocabulary logits tensor is written.
With block width 32, candidate buffers cost 8*N*k/32 bytes, excluding final
merge workspace. Top-1 has the largest traffic saving; higher k reduces it.

This is multiple launches, not a fictitious single-kernel global barrier.
Top-k is selection, not sampling. The API does not implement top-p, repetition
penalties, probabilities, logprobs, or arbitrary logits processors. Top-1 ties
choose the lowest token ID; top-k tied ID ordering may differ from torch.topk.

## Model experiment and honest memory accounting

The `int4_decode` model variant packs every bias-free linear layer, including the
LM head, and uses the candidate only for batch-1 single-token forwards. Dense
prefill stays unchanged. This adapter intentionally retains original weights
for A/B testing, so **resident memory increases**, even though packed buffers
are smaller. A later quantized-only runtime must release dense projection
weights and provide an appropriate prefill path before claiming model savings.

Qwen ties input embeddings and the LM head. Packing the output head alone does
not release the BF16 embedding matrix. The standard Transformers adapter still
returns full logits for generation compatibility; compact argmax/top-k is a
separate custom decode endpoint, not silently substituted into `generate()`.

## Validation gates

```bash
python -m pytest learning/09_int4/test_int4.py -q
python profiling/int4_decode_benchmark.py --output results/int4_decode.json
python profiling/model_workload.py --variant int4_decode --output results/int4_model.json
python profiling/quality_probe.py --variant int4_decode --output results/int4_quality.json
```

The benchmark captures real decode activations for Q, gate, down and LM-head
projections and separates two errors:

1. **Kernel error:** fused result versus FP32 dequantized-weight computation.
2. **Quantization error:** quantized-weight result versus original BF16 weights.

Compact selection is compared against the full INT4 kernel's logits, while
BF16 token agreement is reported separately. Timing excludes quantization and
compilation. Broad prompt evaluation and perplexity are still required; four
smoke prompts cannot certify INT4 quality. Quantization may worsen outlier-heavy
channels or the LM head: measure first, then consider smaller groups, calibration,
or retaining sensitive layers in BF16.
