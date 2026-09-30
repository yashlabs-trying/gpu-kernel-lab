# Contributing

## Development setup

Create a Python 3.10+ environment and install the CPU development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -e ".[cpu,llama]"
python -m pip install fastapi httpx psutil ninja
```

Run `pytest -q -m "not gpu and not integration"` before opening a change. GPU
tests must be marked `gpu`; tests that require a running server or downloaded
weights must be marked `integration`.

## Performance changes

Performance claims require a schema-v2 JSON artifact, a clean result-validator
run, the exact matrix case, environment metadata, and a correctness artifact.
Do not compare unmatched concurrency, token counts, sampling, dtypes, or GPU
environments. A custom kernel is accepted only if it passes quality gates and
beats the active framework path end to end.

Do not commit model weights, access tokens, profiler binaries, or generated
caches. Keep historical results when needed for provenance, but label invalid
methodologies prominently.
