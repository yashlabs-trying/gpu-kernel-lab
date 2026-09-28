#!/usr/bin/env python3
"""Re-save tokenizer config using the installed (older) transformers so
vLLM can load it. Reads the existing tokenizer.json + config and rewrites
tokenizer_config.json with a class the current transformers recognizes."""
import json
import sys

from transformers import AutoTokenizer

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/llama-3.2-3b-hf"

tok = AutoTokenizer.from_pretrained(path)
tok.save_pretrained(path)
print("resaved tokenizer to", path)
print("class:", tok.__class__.__name__)
