# Qwen paged-serving integration — SM89

Date: 2026-09-29  
GPU: NVIDIA RTX 2000 Ada Generation (SM89, 16 GB)  
Software: PyTorch 2.8.0+cu128, Triton 3.4.0, Transformers 5.17.0  
Model: Qwen/Qwen3-0.6B, BF16

## Decision

The correctness-first paged executor is **accepted**. It preserves Qwen's tiled
SDPA prefill, scatters each produced K/V range once into fixed physical pages,
batches variable-length decode requests, reads cache entries through block
tables, and reclaims all blocks at request completion.

The direct Triton paged-attention kernel is **experimental**. It wins its kernel
benchmark but does not pass the model-level 99% decode-argmax gate. Production
defaults therefore retain gather-plus-SDPA.

## Exact executor quality

Independent deterministic workload:

| Check | Result |
|---|---:|
| Prompts | 256 |
| Prefill argmax | **256 / 256 (100%)** |
| Decode argmax | **1,024 / 1,024 (100%)** |
| Maximum absolute logit difference | **0.0** |
| Mean absolute logit difference | **0.0** |
| Minimum logit cosine | 0.999999762 |
| Non-finite values | 0 |
| Cache fully reclaimed | yes |

The four-request end-to-end serving smoke also matched all 16 greedy baseline
tokens exactly, completed all requests without errors, and reclaimed all 36
allocated cache blocks. That smoke validates integration, not tail latency.

## Direct paged-attention candidate

Model-level result over 256 prompts and four decode decisions per prompt:

| Check | Result |
|---|---:|
| Prefill argmax | 100% |
| Decode argmax | **97.65625%** |
| Maximum absolute logit difference | 1.375 |
| Minimum logit cosine | 0.999045 |
| Release gate | **FAIL** (<99% decode agreement) |

Accurate CUDA libdevice exponent/logarithm did not change the 97.65625% agreement
observed in the follow-up sample. The remaining difference is attention reduction
ordering relative to PyTorch SDPA, not cache addressing.

## Attention microbenchmark

Each value is the median of 100 GPU-event repetitions after 20 warmups. The
baseline includes physical-page gather, GQA expansion, and PyTorch SDPA.

| Context | Gather + SDPA | Direct paged | Speedup | Max abs error |
|---:|---:|---:|---:|---:|
| 128 | 177.504 us | 101.728 us | **1.74x** | 0.00390625 |
| 512 | 228.704 us | 102.016 us | **2.24x** | 0.00097656 |
| 2,048 | 286.400 us | 102.976 us | **2.78x** | 0.00048828 |
| 4,096 | 587.472 us | 109.456 us | **5.37x** | 0.00048828 |

These are isolated attention results, not end-to-end serving gains.

## Reproduction

```bash
python profiling/qwen_paged_executor_quality.py \
  --prompts 256 --decode-steps 4 --local-files-only \
  --output results/qwen_paged/exact_quality.json

python profiling/qwen_paged_executor_quality.py \
  --prompts 256 --decode-steps 4 --direct-attention --local-files-only \
  --output results/qwen_paged/direct_quality.json

python profiling/paged_attention_benchmark.py \
  --output results/qwen_paged/attention_benchmark.json

python profiling/qwen_serving_gpu_smoke.py \
  --requests 4 --max-new-tokens 4 --prefill-chunk 32 --local-files-only \
  --output results/qwen_paged/serving_smoke.json
```

## Remaining production work

1. Match SDPA's accepted numerical behavior in the direct block-table kernel.
2. Replace the correctness prefill scatter with a direct paged chunked-prefill
   kernel after an equivalent quality gate exists.
3. Capture stable decode batches in CUDA Graph buckets.
4. Run statistically powered concurrent p50/p90/p99 benchmarks after warmup.
5. Repeat accuracy and performance gates on a second GPU architecture.
