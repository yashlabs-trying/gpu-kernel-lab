# GPU acceptance runbook

Nothing in this runbook may be reported as passed until its artifact exists and
`kernellab validate-result` succeeds where applicable.

## 1. Provision and preflight

```bash
git clone <repository-url> gpu-kernel-lab
cd gpu-kernel-lab
KERNELLAB_CUDA_VARIANT=cu128 bash models/optimized-llama/scripts/setup_env.sh
source .venv-llama/bin/activate
kernellab doctor | tee results/llama-doctor.json
kernellab download llama-3.2-3b
python models/optimized-llama/profiling/export_shapes.py \
  --model /workspace/models/llama-3.2-3b-hf \
  --output results/llama-shapes.json
python models/optimized-llama/scripts/model_integrity.py create \
  /workspace/models/llama-3.2-3b-hf --output results/llama-checksums.json
```

## 2. Matched PyTorch and vLLM measurements

Run PyTorch at each required concurrency and validate every JSON result:

```bash
kernellab benchmark llama-3.2-3b --engine pytorch \
  --model-path /workspace/models/llama-3.2-3b-hf \
  --input-tokens 128 --output-tokens 128 --concurrency 1 \
  --repetitions 100 --output results/pytorch-c1.json
```

Launch vLLM in another terminal, keeping prefix caching disabled:

```bash
kernellab serve llama-3.2-3b \
  --model-path /workspace/models/llama-3.2-3b-hf \
  --max-model-len 8192 --max-num-seqs 64
```

Then benchmark the identical prompt IDs, output length, sampling, dtype, and
concurrency:

```bash
kernellab benchmark llama-3.2-3b --engine vllm \
  --model-path /workspace/models/llama-3.2-3b-hf \
  --input-tokens 128 --output-tokens 128 --concurrency 1 \
  --repetitions 100 --output results/vllm-c1.json
```

Generate the full case list with `kernellab plan --output
results/llama-matrix.json`. Execute concurrency 1/2/4/8/16/32/64, input lengths
8/128/512/2048/8192, and output lengths 32/128/256. Relaunch the server for
eager/graph and FP16/INT8 variants; do not change multiple variables within an
A/B pair.

## 3. Quality gates

Capture FP16 and each candidate across the complete checked-in corpus using one
model load and the same seed/token count:

```bash
python models/optimized-llama/quality/capture_corpus.py \
  --model /workspace/models/llama-3.2-3b-hf \
  --prompts models/optimized-llama/quality/prompts.json \
  --output results/reference-corpus.npz

python models/optimized-llama/quality/evaluate.py \
  --reference results/reference-corpus.npz --candidate results/candidate-corpus.npz \
  --output results/candidate-quality.json
```

Run the pinned external suite with `quality/run_lm_eval.py`; limited runs are
integration smoke tests only, never release evidence. A transparent replacement
requires ≥99% argmax/token
agreement, ≥0.999 minimum per-layer cosine, no tokenizer mismatch, and no NLL
regression above the encoded tolerance.

## 4. Reliability and serving behavior

```bash
python models/optimized-llama/benchmarks/probe_cancellation.py \
  --tokenizer /workspace/models/llama-3.2-3b-hf \
  --output results/cancellation.json

python models/optimized-llama/benchmarks/bench_reliability.py \
  --mode soak --duration-seconds 3600 --concurrency 32 \
  --tokenizer /workspace/models/llama-3.2-3b-hf \
  --output results/soak-1h.json

python models/optimized-llama/benchmarks/bench_reliability.py \
  --mode overload --concurrency 128 --requests 1000 \
  --tokenizer /workspace/models/llama-3.2-3b-hf \
  --output results/overload.json

python models/optimized-llama/benchmarks/bench_mixed_fairness.py \
  --model llama-3.2-3b --tokenizer /workspace/models/llama-3.2-3b-hf \
  --short-input-tokens 128 --long-input-tokens 8192 --concurrency 8 \
  --repetitions-per-class 100 --output results/mixed-fairness.json
```

Review `/metrics` before and after for cache recovery and GPU memory stability.
Normal load must have zero failed requests and no residual KV/cache allocation.

## 5. Analysis, profiling, and multi-GPU

Use `benchmarks/analyze_results.py` followed by `plot_frontier.py` to produce
the throughput/p99 frontier, saturation point, length fairness, and tensor-
parallel scaling efficiency. Capture Nsight Systems traces and active attention
backend logs. Run tensor-parallel sizes 1 and 2 (and 4 if available), save
`nvidia-smi topo -m`, compare TP logits with TP1, and record PCIe/NVLink
communication overhead.

Profile the native HF decode path with `profiling/profile_kernels.py`, record
the selected vLLM attention backend from its launch log, and execute
`kernels/bench_rmsnorm_vs_vllm.py`. A microbenchmark winner is only eligible for
a subsequent end-to-end vLLM integration test; it is not production-accepted.
Measure full process/model readiness with `benchmarks/bench_server_startup.py`
while no other server is bound to the target port.

Only after these artifacts pass may kernel candidates be accepted or a release
tag be created.
