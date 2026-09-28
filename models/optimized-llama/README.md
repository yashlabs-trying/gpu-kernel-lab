# Optimized Llama

Optimizing Llama for GPU inference speed and throughput — and documenting every
layer of *how* GPU optimization works, from bits to billing.

**Target:** Llama-3.2-3B · **Hardware:** NVIDIA A40 (Ampere, 45 GB) ·
**Stack:** vLLM + PyTorch · **Result:** up to **81x throughput** over naive PyTorch.

---

## What we measured (real numbers, this repo)

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

## Key finding

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
├── benchmarks/                    # TTFT / ITL / throughput harness
│   ├── bench_baseline.py          #   naive PyTorch fp16 baseline
│   ├── bench_vllm.py              #   vLLM throughput benchmark
│   └── bench_sweep.py             #   batch × input-len × output-len sweep
├── profiling/                     # Nsight Compute / Systems + kernel profiling
│   ├── profile_decode.py          #   minimal decode target for nsys
│   └── profile_kernels.py         #   torch.profiler fallback (ncu is blocked on RunPod)
├── scripts/                       # setup + model handling (weights never committed)
│   ├── setup_env.sh               #   venv: torch, vLLM, transformers, gguf
│   ├── download_model.sh          #   gated HF download (or use Ollama → GGUF)
│   ├── convert_gguf_to_hf.py      #   GGUF → safetensors (no HF token needed)
│   └── verify_hf.py               #   parity check after conversion
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
bash scripts/setup_env.sh          # venv at /opt/venv (or adjust for your driver)
ln -sf /opt/venv/bin/ninja /usr/local/bin/ninja   # vLLM needs ninja on PATH
```

### 3. Baseline (naive PyTorch)
```bash
/opt/venv/bin/python benchmarks/bench_baseline.py \
    --model-dir /workspace/models/llama-3.2-3b --num-decode 128
```

### 4. Optimized (vLLM)
```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/venv/bin/python benchmarks/bench_vllm.py \
    --model /workspace/models/llama-3.2-3b-hf --quantization none \
    --num-prompts 64 --max-tokens 128

# full sweep:
FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/venv/bin/python benchmarks/bench_sweep.py \
    --model /workspace/models/llama-3.2-3b-hf \
    --batches 1,8,32,64 --input-lens 8,128,1024 --output-lens 64,256
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
