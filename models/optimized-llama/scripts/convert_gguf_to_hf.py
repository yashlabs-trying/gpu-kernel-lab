#!/usr/bin/env python3
"""Convert the fp16 GGUF model to a HuggingFace safetensors checkpoint.

Loads model.gguf via transformers (de-quantizing to fp16) and saves it back
out as config.json + model.safetensors + tokenizer, so vLLM/TensorRT can
consume it directly. No HF token required.

Usage:
  python scripts/convert_gguf_to_hf.py \
      --gguf /workspace/models/llama-3.2-3b/model.gguf \
      --out /workspace/models/llama-3.2-3b-hf
"""
import argparse
import json
from pathlib import Path

import torch
from model_integrity import inventory
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_fast_tokenizer(tokenizer, output: Path) -> None:
    """Save a GGUF-derived tokenizer without copying the GGUF vocabulary file."""
    tokenizer.save_pretrained(output, legacy_format=False)
    # ``fix_mistral_regex`` is a one-time load migration. Some Transformers
    # versions persist it in init_kwargs; retaining it makes the next load try
    # to patch the already-corrected backend and can raise TypeError.
    config_path = output / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.pop("fix_mistral_regex", None) is not None:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    unexpected_vocab = output / "tokenizer.model"
    if unexpected_vocab.exists():
        raise RuntimeError(
            "refusing output containing tokenizer.model copied from the GGUF"
        )


def load_gguf_tokenizer(source: str):
    return AutoTokenizer.from_pretrained(
        source,
        gguf_file="model.gguf",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-checksum", action="store_true")
    args = ap.parse_args()

    tok = load_gguf_tokenizer(args.gguf)
    model = AutoModelForCausalLM.from_pretrained(
        args.gguf,
        gguf_file="model.gguf",
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    # A tokenizer loaded from GGUF keeps the GGUF path in ``vocab_file``.
    # The legacy save path therefore copies the complete multi-GB GGUF as
    # ``tokenizer.model``.  Save the fast-tokenizer JSON only: it is
    # self-contained and is what current Transformers/vLLM consume.
    save_fast_tokenizer(tok, Path(args.out))
    model.save_pretrained(args.out, safe_serialization=True)
    if not args.skip_checksum:
        output = Path(args.out)
        manifest = {"algorithm": "sha256", "files": inventory(output)}
        (output / "CHECKSUMS.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print("saved to", args.out)


if __name__ == "__main__":
    main()
