# Upstream provenance

This directory was imported from:

- Repository: <https://github.com/yashlabs-trying/optimized-llama>
- Branch: `main`
- Imported commit: `80087ed89d02c2b6961728e6a810da0328576bac`
- Import date: 2026-09-28
- Integration location: `models/optimized-llama/`

The snapshot is stored directly in GPU Kernel Lab so ordinary clones are
self-contained and do not require `git submodule update`.

## Updating the snapshot

From the GPU Kernel Lab root:

```bash
git remote get-url optimized-llama >/dev/null 2>&1 || \
  git remote add optimized-llama https://github.com/yashlabs-trying/optimized-llama.git
git fetch optimized-llama main
git diff 80087ed89d02c2b6961728e6a810da0328576bac..optimized-llama/main --
```

Review upstream changes before copying or merging them. Update the pinned commit
above in the same commit as the imported changes. Keep model weights, caches,
virtual environments, and generated profiler binaries out of Git.

## Environment isolation

The Llama project uses a vLLM-oriented software matrix that differs from the
Qwen Triton laboratory. Run its setup script from this directory and do not
install its pinned packages into the Qwen environment:

```bash
cd models/optimized-llama
bash scripts/setup_env.sh
```

The imported README describes two tested CUDA/driver combinations and the exact
benchmark commands.
