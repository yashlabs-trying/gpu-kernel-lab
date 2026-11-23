"""Skip Triton kernel test modules cleanly on machines without Triton.

CPU-only PyTorch wheels do not ship Triton. Kernel tests that need it are
skipped at collection instead of failing the whole CPU suite.
"""
import importlib.util

collect_ignore_glob = []
if importlib.util.find_spec("triton") is None:
    collect_ignore_glob += ["learning/*/test_*.py"]
