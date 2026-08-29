# RunPod execution guide

This guide keeps paid GPU time focused on setup verification, execution, and
measurement. KernelLab code and small JSON results live in GitHub. Public model
weights come from Hugging Face and do not belong in this repository.

## 1. Deploy the Pod

Use a Linux NVIDIA GPU Pod with direct SSH access. For the first KernelLab
baseline, an Ampere GPU such as an A40, A5000, or RTX 3090 is sufficient.

Recommended starting storage:

- Container disk: 20 GB for temporary operating-system files.
- Volume disk: 30–50 GB mounted at `/workspace` if the Pod will be stopped and
  resumed during the same period.
- Do not retain a stopped volume for weeks; storage billing continues.

Set an automatic stop or termination deadline as protection against forgetting
the running Pod.

## 2. Connect GitHub

Add your SSH public key in RunPod and connect to the Pod. Authenticate GitHub on
the Pod without pasting a secret into project files:

```bash
gh auth login --web --git-protocol https
```

Complete the displayed device-login flow in your own browser. Then clone:

```bash
cd /workspace
git clone https://github.com/yashlabs-trying/gpu-kernel-lab.git
cd gpu-kernel-lab
```

Confirm that pushes are authorized:

```bash
gh auth status
git remote -v
```

## 3. Install the environment

Run:

```bash
bash scripts/setup_runpod.sh
```

The script creates `.venv`, installs the pinned PyTorch CUDA build and project
dependencies, and performs a CUDA tensor test. Installation is expected to take
time only on a fresh Pod.

Do not run the benchmark if the environment check reports no CUDA device.

## 4. Download and benchmark Qwen

Qwen3-0.6B is public; a paid Hugging Face account is not required. Run:

```bash
bash scripts/run_baseline.sh
```

This command:

1. Records GPU and software metadata.
2. Downloads `Qwen/Qwen3-0.6B` into the Hugging Face cache.
3. Loads the model in BF16 and performs a short generation.
4. Benchmarks prefill at several sequence lengths.
5. Benchmarks cached autoregressive decode.
6. Writes timestamped JSON files under `results/`.

The first run includes model download and kernel initialization. The benchmark
itself performs warmups before recording measurements.

## 5. Review the results

Check the generated files before committing:

```bash
ls -lh results
git diff -- results
```

Verify that the JSON identifies the expected GPU and that all values are finite
and plausible. Never manually invent or edit performance measurements.

## 6. Commit and push

Configure the commit identity once on the Pod if necessary:

```bash
git config user.name "Yash-labs-opensource"
git config user.email "novalabs0.ai@gmail.com"
```

Commit only the small environment and benchmark reports:

```bash
git add results
git commit -m "Record initial Qwen GPU baseline"
git push origin main
```

Confirm the push succeeded before deleting anything:

```bash
git status --short --branch
git log -1 --oneline
```

The status should show `main...origin/main` without uncommitted files.

## 7. Stop or terminate safely

- Use **Stop** for a short pause when retaining `/workspace` is worth the storage
  charge.
- Use **Terminate** after the results are on GitHub and the model can be
  downloaded again later.
- Termination permanently deletes Pod-local files that are not held in a
  separate network volume.

Before terminating, verify all three conditions:

- The benchmark command completed successfully.
- The required JSON results are visible on GitHub.
- `git status` is clean and synchronized with `origin/main`.

## Troubleshooting boundary

If setup, CUDA validation, model loading, or a benchmark fails, do not repeatedly
change packages at random while the GPU remains billed. Capture the full error,
stop the Pod, and diagnose it before starting another paid session.
