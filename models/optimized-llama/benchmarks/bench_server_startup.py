#!/usr/bin/env python3
"""Measure vLLM process/model startup until its OpenAI API becomes ready."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from methodology import environment_metadata


def probe(url: str, timeout: float = 2.0) -> tuple[bool, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return 200 <= response.status < 300, body
    except (urllib.error.URLError, TimeoutError):
        return False, None


def stop_process_group(process: subprocess.Popen, grace_seconds: float) -> bool:
    if process.poll() is not None:
        return True
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return True
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--shutdown-grace", type=float, default=30.0)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("server command is required after --")
    if args.poll_interval <= 0 or args.timeout <= 0 or args.shutdown_grace <= 0:
        parser.error("timing options must be positive")

    base_url = args.base_url.rstrip("/")
    if probe(base_url + "/health")[0]:
        raise SystemExit(f"a server is already healthy at {base_url}; cold start is invalid")
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment_before = environment_metadata()
    started_at = datetime.now(UTC).isoformat()
    start_ns = time.perf_counter_ns()
    health_ns = None
    models_ns = None
    models_body = None
    error = None
    graceful_shutdown = None
    with args.server_log.open("wb") as log:
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        process = subprocess.Popen(
            args.command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
            creationflags=creation_flags,
        )
        try:
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                return_code = process.poll()
                if return_code is not None:
                    error = f"server exited before readiness with code {return_code}"
                    break
                if health_ns is None and probe(base_url + "/health")[0]:
                    health_ns = time.perf_counter_ns()
                if health_ns is not None:
                    models_ready, body = probe(base_url + "/v1/models")
                    if models_ready:
                        models_ns = time.perf_counter_ns()
                        models_body = body
                        break
                time.sleep(args.poll_interval)
            else:
                error = f"server did not become ready within {args.timeout} seconds"
        finally:
            graceful_shutdown = stop_process_group(process, args.shutdown_grace)

    result = {
        "schema_version": 1,
        "benchmark": "llama_vllm_process_cold_start",
        "started_at": started_at,
        "command": args.command,
        "base_url": base_url,
        "server_log": str(args.server_log),
        "process_to_health_seconds": (
            (health_ns - start_ns) / 1e9 if health_ns is not None else None
        ),
        "process_to_models_ready_seconds": (
            (models_ns - start_ns) / 1e9 if models_ns is not None else None
        ),
        "models_response": models_body,
        "graceful_shutdown": graceful_shutdown,
        "error": error,
        "environment_before": environment_before,
        "environment_after": environment_metadata(),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
