# Reproducibility matrix

| Environment | Exact stack | Repository support | GPU validation |
|---|---|---|---|
| CPU CI | Python 3.10 and 3.12, current CPU PyTorch | Automated on every push/PR | Not applicable |
| CUDA 12.8 | PyTorch 2.9.0, vLLM 0.13.0, Transformers 4.57.6 | Lockfile, setup scripts, Dockerfile | Pending fresh schema-v2 run |
| CUDA 13.0 | PyTorch 2.13.0, vLLM 0.28.0, Transformers 5.0.0 | Separate lockfile, setup scripts, Dockerfile | Pending fresh schema-v2 run |
| NVIDIA Ampere | A40 target | Benchmark/quality/profile matrix ready | Pending |
| Second architecture | To be selected during GPU phase | Portable fallback required | Pending |
| Tensor parallel 2 | vLLM launch option exposed | Matrix ready | Pending scaling/logit validation |
| Tensor parallel 4 | vLLM launch option exposed | Hardware permitting | Pending |

“Repository support” means the configuration is encoded and CPU-validated. It
does not mean a container image, dependency combination, performance number, or
quality gate has passed on a GPU. Every release report must replace “Pending”
with an artifact path and hardware metadata rather than an unsupported claim.
