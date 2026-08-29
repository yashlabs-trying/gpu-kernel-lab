#!/usr/bin/env python3
"""Download the public KernelLab model into the Hugging Face cache."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    path = snapshot_download(repo_id=args.model, revision=args.revision)
    print(f"Downloaded {args.model} to {path}")


if __name__ == "__main__":
    main()
