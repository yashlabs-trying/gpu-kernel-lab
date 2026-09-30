# Llama deployment

## NVIDIA Container Toolkit

Install a compatible NVIDIA driver and NVIDIA Container Toolkit on the host,
then verify `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04
nvidia-smi` before building KernelLab. The container does not bundle a driver.

```bash
export KERNELLAB_MODEL_ROOT=/absolute/path/to/models
docker compose -f docker-compose.llama.yml build
docker compose -f docker-compose.llama.yml up
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/metrics
```

The model volume is read-only. Stop with `docker compose -f
docker-compose.llama.yml down`; the two-minute grace period lets vLLM drain and
release workers. Adjust context and memory with `kernellab serve` options.

CUDA 12.8 is the primary recorded environment. CUDA 13.0 has a separate lock
file and Dockerfile because its PyTorch/vLLM matrix differs; both require fresh
GPU validation before a release is tagged.
