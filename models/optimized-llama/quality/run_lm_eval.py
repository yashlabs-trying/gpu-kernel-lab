#!/usr/bin/env python3
"""Run a pinned LM Evaluation Harness suite against a local HF checkpoint."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TASKS = ("hellaswag", "arc_easy", "piqa", "winogrande", "wikitext")


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.executable,
        "run",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={args.model}",
        f"dtype={args.dtype}",
        "trust_remote_code=False",
        "--tasks",
        *args.tasks,
        "--device",
        args.device,
        "--batch_size",
        args.batch_size,
        "--seed",
        str(args.seed),
        "--output_path",
        str(args.output_dir),
        "--log_samples",
    ]
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standard LM tasks; do not use limited runs as release evidence."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", default="auto:4")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit",
        type=float,
        help="Smoke tests only: integer examples or a fraction in (0, 1].",
    )
    parser.add_argument("--executable", default="lm_eval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    manifest = {
        "schema_version": 1,
        "lm_eval_requirement": "lm-eval==0.4.13",
        "created_at": datetime.now(UTC).isoformat(),
        "model": str(args.model.resolve()),
        "tasks": args.tasks,
        "seed": args.seed,
        "limit": args.limit,
        "limited_run_is_release_evidence": False if args.limit is not None else None,
        "command": command,
    }
    if args.dry_run:
        print(shlex.join(command))
        return 0
    if not args.model.exists():
        raise SystemExit(f"model path does not exist: {args.model}")
    if shutil.which(args.executable) is None:
        raise SystemExit(
            f"{args.executable!r} was not found; install requirements-eval.lock"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "kernellab-eval-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    completed = subprocess.run(command, check=False)
    manifest["exit_code"] = completed.returncode
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
