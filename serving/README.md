# KernelLab serving control plane

This package is the model-independent control plane for the next serving phase.
It does not pretend that a CPU scheduler is a finished GPU server: the executor
that connects each plan to the Qwen prefill and paged-attention kernels is the
next integration boundary.

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

## Next GPU integration

1. Change the split-KV kernel from contiguous `[B,H,C,D]` addressing to physical
   block-table lookup.
2. Add a mixed prefill/decode executor consuming `StepPlan`.
3. Capture one graph per supported batch/sequence bucket with inactive-slot masks.
4. Add streaming API and cancellation around the scheduler.
5. Stress-test block reuse, concurrent arrivals, OOM admission, and tail latency.
