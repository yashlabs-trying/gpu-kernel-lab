# Qwen Phase 5/6: portable decode and concurrent serving

Date: 2026-09-29  
Model: `Qwen/Qwen3-0.6B`  
GPU: NVIDIA RTX A4000, 16 GiB, SM86  
Runtime: PyTorch 2.13.0+cu130, BF16

## Release decision

The single-GPU Qwen serving core is accepted on SM86. The same exact serving
path was previously validated on SM89, and uncalibrated GPU architectures now
select a conservative exact fallback instead of inheriting another GPU's tuning.

Direct paged attention remains excluded by request and remains disabled because
its quality gate failed. This phase does not claim a distributed tensor-parallel
runtime; the portability preflight reports peer topology and explicitly labels
multi-GPU tensor parallelism as requiring separate validation.

## Decode changes

- Removed per-token GPU-to-CPU synchronizations for KV lengths and slot checks.
- Passed fixed graph-bucket sequence widths into production KV metadata.
- Corrected the engine to execute latency-sensitive decode before admitted
  chunked-prefill work.
- Added an adaptive 96-token prefill cap while decode requests are active.
  Prefill-only traffic may still use the larger configured chunk.

## Numerical gate after the changes

| Check | Result |
|---|---:|
| Prefill argmax agreement | **100.000%** (256/256) |
| Decode argmax agreement | **100.000%** (1,024/1,024) |
| Maximum/mean logit difference | **0.0 / 0.0** |
| Minimum cosine similarity | **0.99999976** |
| Cache reclamation | pass |

This rerun used the shared pod's newer PyTorch 2.13/CUDA 13 environment. The
unchanged exact result is additional runtime-portability evidence; it is not
mixed into the older PyTorch 2.8 latency comparison.

## Matched 16-request serving comparison

Workload: sixteen simultaneous requests; prompt lengths cycle through 32, 64,
128, and 256 tokens; sixteen generated tokens each; maximum decode batch eight.
Tokenization and HTTP transport are excluded.

| Metric | Original batch-8 order | Adaptive decode-first | Change |
|---|---:|---:|---:|
| Generated throughput | 57.30 tok/s | **65.06 tok/s** | **+13.53%** |
| Wall time | 4.467 s | **3.935 s** | **-11.92%** |
| TTFT p50 | 2,358.99 ms | **1,884.71 ms** | **-20.11%** |
| ITL p50 | 50.77 ms | **40.19 ms** | **-20.85%** |
| ITL p90 | 180.17 ms | **150.84 ms** | **-16.28%** |
| ITL p99 | 674.53 ms | **500.52 ms** | **-25.80%** |

The 96-token cap was selected from measured 64/96/128-token candidates because
it improved throughput and every reported latency percentile together.

For context, serial batch-1 produced 27.44 tok/s, so adaptive continuous batching
delivered **2.37x** the generated-token throughput on this workload.

## Larger concurrent run

The accepted configuration completed 64 simultaneous requests:

| Metric | Result |
|---|---:|
| Completed / failed / rejected | **64 / 0 / 0** |
| Generated tokens | 1,024 |
| Generated throughput | **65.52 tok/s** |
| ITL p50 / p90 / p99 | **40.57 / 182.35 / 407.29 ms** |
| TTFT p50 / p90 / p99 | 8.136 / 14.153 / 14.937 s |
| KV blocks reclaimed | **544/544** |

The TTFT values include queueing from submitting all 64 requests simultaneously
to one 16 GiB GPU. They should not be read as isolated-request TTFT.

## Portability preflight

`scripts/check_qwen_portability.py` loads only the model configuration and reports:

- exact model signature;
- CUDA device name, compute capability, SM count, warp size, and memory;
- selected calibrated policy or exact fallback per GPU;
- peer-access topology and a safe deployment recommendation.

On this pod it selected `ampere-sm86-lossless-paged-gather`. SM89 has its own
checked-in accepted profile. Unknown model shapes and GPU architectures remain
runnable through the exact fallback.

## Reproduce

```bash
python -m pytest serving -q

python scripts/check_qwen_portability.py --local-files-only \
  --output results/qwen_phase56_a4000/portability.json

python profiling/qwen_serving_load.py \
  --requests 64 --prompt-lengths 32 64 128 256 --output-tokens 16 \
  --max-batch-size 8 --prefill-chunk 128 --prefill-budget 512 \
  --decode-prefill-chunk 96 --local-files-only \
  --output results/qwen_phase56_a4000/load_64_b8_cap96.json
```

Phase 7/direct-attention replacement was intentionally not performed.
