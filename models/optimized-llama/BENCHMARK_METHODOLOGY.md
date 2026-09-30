# Llama benchmark methodology

This document is the release contract for Llama performance claims. Results
created by the older benchmark scripts before schema version 2 are historical
and must not be used for production comparisons.

## Metric definitions

- **TTFT:** monotonic time from client request submission to receipt of the
  first generated token. PyTorch wall timing synchronizes CUDA before the start
  and at the first token boundary.
- **ITL:** monotonic gaps between consecutive token receipts for one request.
  It is never computed as wall time divided by token count.
- **GPU-only latency:** CUDA-event time, reported separately from server latency.
- **Input throughput:** successfully processed prompt tokens divided by measured
  wall time.
- **Output throughput:** generated tokens divided by measured wall time. Prompt
  tokens are never included in decode throughput.
- **End-to-end latency:** request submission through completion at the client.

All reports contain raw timestamps, token counts, warmup/repetition counts,
sampling settings, prompt token IDs, and before/after environment snapshots.
P99 is null unless at least 100 measured requests were run.

## Required comparison controls

A comparison is valid only when model weights, prompt token IDs, requested
output length, sampling, dtype/quantization, concurrency, GPU, software stack,
and server settings match. Batch/concurrency 64 throughput must not be compared
to batch-1 latency as if it were the same workload.

Start vLLM with prefix caching disabled so repeated deterministic prompts do
not receive an unintended cache advantage:

```bash
vllm serve /workspace/models/llama-3.2-3b-hf \
  --served-model-name llama-3.2-3b \
  --dtype float16 \
  --disable-prefix-caching
```

Run the matched batch-1 pair:

```bash
python benchmarks/bench_torch_baseline.py \
  --model-dir /workspace/models/llama-3.2-3b-hf \
  --input-tokens 128 --output-tokens 128 \
  --warmups 5 --repetitions 100 \
  --output results/pytorch-b1.json

python benchmarks/bench_serving.py \
  --model llama-3.2-3b \
  --tokenizer /workspace/models/llama-3.2-3b-hf \
  --input-tokens 128 --output-tokens 128 --concurrency 1 \
  --warmups 5 --repetitions 100 \
  --output results/vllm-c1.json
```

Run the vLLM concurrency sweep through the same streaming runner once per
concurrency (`1, 2, 4, 8, 16, 32, 64`). The PyTorch runner accepts the same
`--concurrency` argument and uses one CUDA stream per active request. Do not
substitute batch size or extrapolate from batch 1.

For CUDA Graph A/B, keep every argument fixed and compare a server launched
normally with one launched using `--enforce-eager`. FP16/INT8 comparisons follow
the same rule and use separate server launches.

`bench_sweep.py` and `bench_vllm.py` use vLLM's synchronous offline API. They
therefore report throughput only. Streaming TTFT and ITL come exclusively from
`bench_serving.py`, which requests `return_token_ids` and timestamps each delta
token as it reaches the client. The server must stream one token per chunk
(stream interval 1); the benchmark fails instead of inventing per-token timing
when a chunk contains multiple tokens.
