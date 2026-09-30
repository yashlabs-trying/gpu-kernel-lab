#!/usr/bin/env python3
"""Create/verify model checksums and compare tokenizer behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TRACKED_PATTERNS = (
    "*.json",
    "*.model",
    "*.safetensors",
    "*.txt",
)
IGNORED_FILES = {"CHECKSUMS.json", "tokenizer_parity.json"}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(directory: Path) -> dict[str, str]:
    files: set[Path] = set()
    for pattern in TRACKED_PATTERNS:
        files.update(
            path
            for path in directory.rglob(pattern)
            if path.is_file() and path.name not in IGNORED_FILES
        )
    return {
        path.relative_to(directory).as_posix(): hash_file(path)
        for path in sorted(files)
    }


def tokenizer_parity(reference: Path, candidate: Path, prompts: list[str]) -> dict:
    from transformers import AutoTokenizer

    left = AutoTokenizer.from_pretrained(reference)
    right = AutoTokenizer.from_pretrained(candidate)
    cases = []
    for text in prompts:
        left_ids = left(text, add_special_tokens=True).input_ids
        right_ids = right(text, add_special_tokens=True).input_ids
        cases.append(
            {
                "text": text,
                "match": left_ids == right_ids,
                "reference": left_ids,
                "candidate": right_ids,
            }
        )
    chat_cases = []
    if hasattr(left, "apply_chat_template") and hasattr(right, "apply_chat_template"):
        for text in prompts:
            messages = [{"role": "user", "content": text}]
            try:
                left_chat = left.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                right_chat = right.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            except (ValueError, TypeError):
                break
            chat_cases.append(
                {
                    "text": text,
                    "match": left_chat == right_chat,
                    "reference": left_chat,
                    "candidate": right_chat,
                }
            )
    special_token_ids_match = all(
        getattr(left, attribute) == getattr(right, attribute)
        for attribute in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    )
    chat_templates_match = all(case["match"] for case in chat_cases)
    return {
        "vocabulary_size_match": len(left) == len(right),
        "special_tokens_match": left.special_tokens_map == right.special_tokens_map,
        "special_token_ids_match": special_token_ids_match,
        "cases": cases,
        "chat_template_cases": chat_cases,
        "passed": len(left) == len(right)
        and left.special_tokens_map == right.special_tokens_map
        and special_token_ids_match
        and all(case["match"] for case in cases)
        and chat_templates_match,
        "chat_templates_match": chat_templates_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("directory", type=Path)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--manifest", type=Path, required=True)
    parity = subparsers.add_parser("tokenizer-parity")
    parity.add_argument("reference", type=Path)
    parity.add_argument("candidate", type=Path)
    parity.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "create":
        payload = {"algorithm": "sha256", "files": inventory(args.directory)}
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.output)
        return
    if args.command == "verify":
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))["files"]
        actual = inventory(args.directory)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            changed = sorted(
                key
                for key in set(actual) & set(expected)
                if actual[key] != expected[key]
            )
            raise SystemExit(
                f"checksum mismatch: missing={missing}, "
                f"unexpected={unexpected}, changed={changed}"
            )
        print("checksums valid")
        return

    prompts_path = Path(__file__).resolve().parents[1] / "quality" / "prompts.json"
    prompts = [item["text"] for item in json.loads(prompts_path.read_text(encoding="utf-8"))]
    result = tokenizer_parity(args.reference, args.candidate, prompts)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if not result["passed"]:
        raise SystemExit("tokenizer parity failed")


if __name__ == "__main__":
    main()
