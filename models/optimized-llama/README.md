# Optimized Llama

Optimizing Llama for GPU inference speed and throughput — and documenting every
layer of *how* GPU optimization works, from bits to billing.

**Target:** Llama-3.2-3B · **Hardware:** NVIDIA A40 (Ampere, 45 GB) ·
**Stack:** vLLM + PyTorch · **Status:** benchmark methodology repaired; fresh
matched production measurements are pending.

---

## Production-readiness status

The original benchmark mixed prompt and output tokens in throughput, derived
an approximate per-request ITL from aggregate wall time, compared high-batch
throughput with batch-1 latency, and emitted zero-valued vLLM latency fields.
Those claims are invalid for production comparison. Schema-v2 runners now use
streamed token timestamps, distinct input/output throughput, synchronized CUDA
events, deterministic token IDs, warmups, raw samples, environment telemetry,
and a 100-request minimum for p99. See
[`BENCHMARK_METHODOLOGY.md`](BENCHMARK_METHODOLOGY.md).

## Historical measurements (invalidated; do not cite as production results)

### Baseline — naive PyTorch fp16, single stream
| Metric | Value |
|---|---|
| TTFT (prefill) | 236 ms |
| ITL (inter-token latency) | 22.7 ms |
| Decode throughput | **44 tok/s** |
| GPU utilization | 62% SM, 56% mem |

### vLLM sweep — batch × sequence length
| Batch | out=256 tok/s | in=1024, out=256 tok/s | vs 44 tok/s baseline |
|---|---|---|---|
| 1 | 75 | 360 | 1.7x |
| 8 | 564 | 2565 | 12.8x |
| 32 | 2013 | 8525 | 46x |
| 64 | **3588** | **15292** | **81x** |

Full matrix: [`results/sweep.md`](results/sweep.md) · Baseline analysis:
[`results/baseline.md`](results/baseline.md).

---

The tables below are retained only for provenance while fresh schema-v2 runs
are collected. In particular, the **81x** figure compares batch-64 aggregate
throughput with a batch-1 baseline and is not a matched speedup.

## Historical hypothesis to revalidate

LLM **decode** is memory-bound (1 FLOP/byte — it reads all weights to emit one
token). The single biggest lever is **continuous batching**, not quantization:
it saturates the bandwidth a single stream can't touch. Quantization's speedup
is biggest for *single-stream* decode; batched throughput wins come from
batching + kernels + CUDA graphs.

---

## Project structure

```
optimized-llama/
├── ULTIMATE_GPU_OPTIMIZATION.md   # The full guide: bits → billing (8 parts)
├── BENCHMARK_METHODOLOGY.md       # Schema-v2 metric and comparison contract
├── PRODUCTION_READINESS.md        # Implemented versus GPU-validated status
├── KERNEL_PROVENANCE.md           # Local work versus framework kernels
├── benchmarks/                    # TTFT / ITL / throughput harness
│   ├── bench_baseline.py          #   naive PyTorch fp16 baseline
│   ├── bench_vllm.py              #   vLLM throughput benchmark
│   ├── bench_sweep.py             #   batch × input-len × output-len sweep
│   ├── bench_mixed_fairness.py    #   simultaneous short/long scheduler load
│   └── bench_server_startup.py    #   process/model cold-start timing
├── profiling/                     # Nsight Compute / Systems + kernel profiling
│   ├── profile_decode.py          #   minimal decode target for nsys
│   └── profile_kernels.py         #   torch.profiler fallback (ncu is blocked on RunPod)
├── scripts/                       # setup + model handling (weights never committed)
│   ├── setup_env.sh               #   venv: torch, vLLM, transformers, gguf
│   ├── download_model.sh          #   gated HF download (or use Ollama → GGUF)
│   ├── convert_gguf_to_hf.py      #   GGUF → safetensors (no HF token needed)
│   ├── model_integrity.py         #   checksums and tokenizer parity
│   └── verify_hf.py               #   parity check after conversion
├── quality/                       # deterministic captures and quality gates
├── Dockerfile.cu128               # primary recorded container stack
├── Dockerfile.cu130               # alternate recorded container stack
└── results/                       # measured baseline + sweep + optimization reports
```

## The two hardware situations we hit

1. **Driver 580 / CUDA 13.0** → torch `2.13+cu130`, vLLM `0.28`.
2. **Driver 570 / CUDA 12.8** (newer pod) → torch `2.9+cu128`, vLLM `0.13`,
   transformers `<5`, `FLASHINFER_DISABLE_VERSION_CHECK=1`.

The repo runs on both; see `scripts/` for the working versions.

---

## How to reproduce

### 1. Get the model (one of two ways)

**A. Gated HuggingFace** (needs a token):
```bash
HF_TOKEN=... bash scripts/download_model.sh
```

**B. Ollama → GGUF → safetensors** (no token, what we used):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & sleep 5
ollama pull llama3.2:3b-text-fp16
# copy the 6.4GB blob to /workspace/models/llama-3.2-3b/model.gguf
/opt/venv/bin/python scripts/convert_gguf_to_hf.py \
    --gguf /workspace/models/llama-3.2-3b \
    --out  /workspace/models/llama-3.2-3b-hf
```

### 2. Set up environment
```bash
KERNELLAB_VENV=/opt/venv KERNELLAB_CUDA_VARIANT=cu128 bash scripts/setup_env.sh
source /opt/venv/bin/activate
kernellab doctor
```

### 3. Matched batch-1 PyTorch baseline
```bash
/opt/venv/bin/python benchmarks/bench_baseline.py \
    --model-dir /workspace/models/llama-3.2-3b-hf \
    --input-tokens 128 --output-tokens 128 \
    --warmups 5 --repetitions 100 \
    --output results/pytorch-b1.json
```

### 4. Matched vLLM streaming benchmark

`kernellab serve` supplies a bundled Llama 3 message-format template so the
OpenAI chat route is operational. The registry still uses the base checkpoint;
formatting does not make it instruction-tuned, so use instruct weights when
conversational quality is required.

```bash
vllm serve /workspace/models/llama-3.2-3b-hf \
    --served-model-name llama-3.2-3b --dtype float16 \
    --no-enable-prefix-caching

/opt/venv/bin/python benchmarks/bench_serving.py \
    --model llama-3.2-3b \
    --tokenizer /workspace/models/llama-3.2-3b-hf \
    --input-tokens 128 --output-tokens 128 --concurrency 1 \
    --warmups 5 --repetitions 100 \
    --output results/vllm-c1.json

# Throughput-only offline sweep (does not report TTFT/ITL):
FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/venv/bin/python benchmarks/bench_sweep.py \
    --model /workspace/models/llama-3.2-3b-hf \
    --concurrencies 1,2,4,8,16,32,64 \
    --input-lens 8,128,512,2048,8192 --output-lens 32,128,256 \
    --output results/vllm-throughput-sweep.json
```

## Profiling

| Tool | Status | Captures |
|------|--------|----------|
| Nsight Systems (`nsys`) | ✅ works | CPU/GPU timeline, launch gaps, batching |
| Nsight Compute (`ncu`) | ❌ `ERR_NVGPUCTRPERM` on RunPod | per-kernel SM/DRAM (needs `CAP_SYS_ADMIN`) |
| `torch.profiler` | ✅ fallback | per-kernel CUDA time |
| `nvidia-smi dmon` | ✅ works | live SM% / mem% / power |

> On containerized cloud GPUs, `ncu` needs perf-counter access the container
> lacks. Use `nsys` + `torch.profiler` as the practical fallback.

## Standard language-model evaluation

The external evaluation environment is isolated so its dataset dependencies do
not perturb the serving lockfile. A limited run is only a download/integration
smoke test and is never release evidence.

```bash
/opt/venv/bin/pip install -r requirements-eval.lock
/opt/venv/bin/python quality/run_lm_eval.py \
    --model /workspace/models/llama-3.2-3b-hf \
    --output-dir results/lm-eval-fp16 \
    --tasks hellaswag arc_easy piqa winogrande wikitext
```

The deterministic kernel/quantization gate uses the checked-in multi-domain
corpus and compares every generated token, logit row, and hidden-state layer:

```bash
/opt/venv/bin/python quality/capture_corpus.py \
    --model /workspace/models/llama-3.2-3b-hf \
    --prompts quality/prompts.json --output results/quality-fp16.npz
/opt/venv/bin/python quality/capture_corpus.py \
    --model /workspace/models/llama-3.2-3b-hf \
    --prompts quality/prompts.json --quantization bitsandbytes-int8 \
    --output results/quality-int8.npz
/opt/venv/bin/python quality/evaluate.py \
    --reference results/quality-fp16.npz --candidate results/quality-int8.npz \
    --output results/quality-int8-vs-fp16.json
```

## Models are never committed

Weights live on the GPU workspace only (`/workspace/models/`) and are
`.gitignore`d. GitHub holds code + docs + results only.

## The full guide

`ULTIMATE_GPU_OPTIMIZATION.md` is the companion deep-dive: 8 parts covering
GPU fundamentals, memory-bound vs compute-bound, what breaks, LLM inference,
the optimization stack + trade-offs, multi-architecture portability,
multi-GPU, and a freelancing playbook.

## License

MIT
