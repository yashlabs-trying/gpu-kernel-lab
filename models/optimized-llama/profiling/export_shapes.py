#!/usr/bin/env python3
"""Export every Llama layer's static dimensions without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernellab.llama_shapes import shape_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    manifest = shape_manifest(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
