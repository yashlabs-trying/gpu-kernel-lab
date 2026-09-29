# KernelLab serving control plane

This package combines a model-independent control plane with a correctness-first
Qwen GPU executor. `QwenPagedExecutor` writes K/V directly into scheduler-owned
physical pages and resolves attention inputs through each request's block table.
Its current gather-plus-SDPA attention is the integration baseline, not the final
direct paged-attention kernel.

## Components

- `PagedKVAllocator` owns fixed physical K/V blocks, per-request logical block
  tables, valid lengths, atomic reservation, and immediate block reuse.
- `ContinuousBatchScheduler` prioritizes ready decode work while admitting one
  bounded prefill chunk per waiting request in round-robin order. Every plan
  reserves and returns physical `(block, offset)` slots before GPU execution.
- `CUDAGraphBuckets` selects the smallest registered `(batch,sequence)` bucket
  and enforces fixed input signatures during capture/replay.
- `Sampler` implements deterministic temperature, top-k, top-p, repetition
  penalty, stop-token, and optional top-logprob processing.

The attention kernel must consume `BatchKVMetadata.block_tables` and `lengths`
directly. It should never construct a dense growing causal mask: for request `b`,
only logical KV positions below `lengths[b]` are valid, and block-table entry
`logical_position // block_size` resolves the physical block.

## Invariants

- A block belongs to at most one live request.
- Reservation either succeeds completely or changes nothing.
- Finished, cancelled, and failed requests release every block immediately.
- Long prompts receive bounded chunks and return to the end of the prefill queue.
- Decode-ready requests are planned before prefill work.
- Graph buckets pad inactive batch slots only; KV lengths prevent inactive or
  invalid cache reads.
- Random generators are per request, so batching order does not change sampling.
- Final-prefill logits produce the first generated token without reserving or
  writing a duplicate cache position. Every later decode item carries the prior
  sampled token and its exact physical destination slot.

## HTTP surface

`create_app(engine, tokenizer)` exposes:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `DELETE /v1/requests/{request_id}`
- `GET /v1/models`
- `GET /health`
- `GET /metrics`

Completion endpoints support ordinary JSON responses and OpenAI-style SSE
streaming. Disconnecting a stream cancels its request and reclaims KV blocks.
Admission failures return HTTP 503 with `Retry-After`; schema errors use FastAPI's
HTTP 422 response. Executor failures become terminal request events, release
memory, and leave the serving loop alive for subsequent work.

## Qwen executor boundary

The initial real executor is deliberately conservative:

- exact eager Qwen layer math;
- exact tiled SDPA prefill with each new K/V range scattered once into pages;
- serial request execution in the initial integration baseline;
- direct writes into `[layer, block, kv_head, block_token, head_dim]`;
- variable-length lookup through `BatchKVMetadata.block_tables`;
- no dense growing mask and no cache concatenation;
- gather-plus-SDPA as the correctness oracle.

This gives the direct paged Triton/CUDA kernel a trustworthy A/B target. It is
not advertised as a throughput winner.

## Next GPU optimization

1. Replace gather-plus-SDPA with GQA-aware physical block-table reads inside the
   split-KV online-softmax kernel.
2. Batch decode requests and implement a tiled, causal chunked-prefill path.
3. Capture one graph per supported batch/sequence bucket with inactive-slot masks.
4. Run dynamic-cache versus paged-executor token/logit acceptance on the GPU.
5. Measure concurrent p50/p90/p99 and reclaim/cancellation behavior under load.
