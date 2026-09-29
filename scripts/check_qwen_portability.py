"""Print a no-weights Qwen CUDA compatibility and kernel-policy report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from transformers import AutoConfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.portability import inspect_qwen_host


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B')
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=AutoConfig.from_pretrained(args.model,local_files_only=args.local_files_only)
    report=inspect_qwen_host(config)
    text=json.dumps(report,indent=2)+'\n'
    print(text,end='')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    raise SystemExit(0 if report['safe_to_start'] else 2)


if __name__=='__main__': main()
