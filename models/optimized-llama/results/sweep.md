# Sweep Results — vLLM vs PyTorch baseline

Hardware: NVIDIA A40 (Ampere, 45 GB, ~696 GB/s), 96 cores.
Model: Llama-3.2-3B (fp16). Driver 570.195.03 (CUDA 12.8).
Stack: vLLM 0.13.0, torch 2.9.0+cu128, transformers 4.57.6.
Date: 2026-09-05.

## Baseline (naive PyTorch fp16, single stream)

| Metric | Value |
|--------|-------|
| TTFT (prefill) | 236.5 ms |
| Mean ITL (decode) | 22.67 ms |
| Decode throughput | 44.11 tok/s |
| End-to-end throughput | 41.08 tok/s |

## vLLM sweep (batch x input_len x output_len)

| Batch | Input len | Output len | Throughput (tok/s) | req/s | approx ITL (ms) |
|-------|-----------|------------|--------------------|-------|-----------------|
| 1 | 8 | 64 | 78.8 | 1.09 | 14.27 |
| 1 | 8 | 256 | 75.5 | 0.29 | 13.65 |
| 1 | 128 | 64 | 218.7 | 1.14 | 13.72 |
| 1 | 128 | 256 | 110.3 | 0.29 | 13.59 |
| 1 | 1024 | 64 | 1162.4 | 1.07 | 14.62 |
| 1 | 1024 | 256 | 360.2 | 0.28 | 13.88 |
| 8 | 8 | 64 | 608.8 | 8.46 | 1.85 |
| 8 | 8 | 256 | 563.8 | 2.14 | 1.83 |
| 8 | 128 | 64 | 1625.1 | 8.46 | 1.85 |
| 8 | 128 | 256 | 815.2 | 2.12 | 1.84 |
| 8 | 1024 | 64 | 8256.6 | 7.59 | 2.06 |
| 8 | 1024 | 256 | 2564.6 | 2.00 | 1.95 |
| 32 | 8 | 64 | 2207.1 | 30.65 | 0.51 |
| 32 | 8 | 256 | 2012.9 | 7.62 | 0.51 |
| 32 | 128 | 64 | 5570.7 | 29.01 | 0.54 |
| 32 | 128 | 256 | 2787.4 | 7.26 | 0.54 |
| 32 | 1024 | 64 | 26881.0 | 24.71 | 0.63 |
| 32 | 1024 | 256 | 8525.2 | 6.66 | 0.59 |
| 64 | 8 | 64 | 3925.7 | 54.52 | 0.29 |
| 64 | 8 | 256 | 3588.1 | 13.59 | 0.29 |
| 64 | 128 | 64 | 9942.5 | 51.78 | 0.30 |
| 64 | 128 | 256 | 4970.1 | 12.94 | 0.30 |
| 64 | 1024 | 64 | 48607.7 | 44.68 | 0.35 |
| 64 | 1024 | 256 | 15291.6 | 11.95 | 0.33 |

## Key observations

1. **Throughput scales strongly with batch and input length.**
   - Decode-bound (short input, long output) is memory-bound: ~75 tok/s at
     batch 1, ~3588 tok/s at batch 64 (batch 8 -> 64 gives ~6x).
   - Prefill-bound (long input) is compute-bound and much faster: input 1024
     reaches 48,608 tok/s at batch 64.

2. **vs baseline (44 tok/s decode)**:
   - batch 8, output 256: 564 tok/s = 12.8x
   - batch 64, output 256: 3588 tok/s = 81x
   - batch 64, input 1024: 48,608 tok/s = 1103x (prefill-dominated)

3. **ITL drops with batch** (memory bandwidth amortized over more streams):
   14.3 ms (batch 1) -> 0.29 ms (batch 64), a ~50x improvement.

4. Chunked prefill is active (max_num_batched_tokens=8192), keeping long
   prompts from stalling decode.

## Reproduce

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 /opt/venv/bin/python benchmarks/bench_sweep.py \
  --model /workspace/models/llama-3.2-3b-hf --quantization none \
  --batches 1,8,32,64 --input-lens 8,128,1024 --output-lens 64,256
```
