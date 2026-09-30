# Architecture

```mermaid
flowchart LR
    CLI[kernellab CLI] --> Registry[Model registry]
    CLI --> Download[Model download and integrity]
    CLI --> Bench[Schema-v2 benchmarks]
    CLI --> Serve[vLLM OpenAI server]
    Bench --> Torch[Transformers batch-1]
    Bench --> Stream[Streaming server client]
    Torch --> Raw[Raw timestamps and CUDA events]
    Stream --> Raw
    Raw --> Validator[Result schema validator]
    Quality[Quality capture] --> Gates[Token/logit/layer/NLL gates]
    Validator --> Reports[Reproducible reports]
    Gates --> Reports
    Serve --> Health[/health]
    Serve --> Models[/v1/models]
    Serve --> Metrics[/metrics]
```

The CLI and registry are CPU-safe and import no GPU framework. GPU benchmark
runners are separate processes so a failed model load or out-of-memory error
has an explicit nonzero exit. vLLM owns production request scheduling, paged KV
cache, streaming, cancellation on disconnect, and graceful process shutdown.

Every result flows through one schema boundary. Raw request and token samples
are authoritative; percentile summaries are derived data. Quality captures
store generated IDs, full next-token logits, and final-position hidden states
for every layer so optimized paths can be compared with the FP16 reference.
