#!/usr/bin/env python3
"""Disconnect streamed requests after the first token and verify server health."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from bench_serving import exact_prompt_ids
from transformers import AutoTokenizer


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def cancel_after_first_token(endpoint: str, payload: dict) -> bool:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=300)
    try:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:].strip())
            choices = chunk.get("choices") or []
            if choices and choices[0].get("token_ids"):
                return True
        return False
    finally:
        response.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="llama-3.2-3b")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt = exact_prompt_ids(tokenizer, "The capital of France is", 128)
    base = args.base_url.rstrip("/")
    metrics_before = fetch(base + "/metrics")
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": 256,
        "min_tokens": 256,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "return_token_ids": True,
    }
    observations = [
        {
            "request": index,
            "received_first_token": cancel_after_first_token(
                base + "/v1/completions", payload
            ),
        }
        for index in range(args.requests)
    ]
    time.sleep(args.settle_seconds)
    result = {
        "schema_version": 1,
        "benchmark": "llama_vllm_cancellation",
        "model": args.model,
        "requests": args.requests,
        "observations": observations,
        "health_after": fetch(base + "/health"),
        "metrics_before": metrics_before,
        "metrics_after": fetch(base + "/metrics"),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if not all(item["received_first_token"] for item in observations):
        raise SystemExit("one or more cancellation probes failed before the first token")


if __name__ == "__main__":
    main()
