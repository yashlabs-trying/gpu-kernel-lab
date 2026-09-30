"""The ``kernellab`` command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .hardware import detect_hardware, print_hardware
from .matrix import build_matrix
from .registry import MODELS, get_model
from .results import ResultValidationError, validate_result_file

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "models" / "optimized-llama"
DEFAULT_LLAMA_CHAT_TEMPLATE = LLAMA / "chat_templates" / "llama3-base.jinja"


def _model_path(name: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = Path(os.environ.get("KERNELLAB_MODEL_ROOT", "/workspace/models"))
    return root / get_model(name).local_directory


def _run(command: list[str], dry_run: bool) -> int:
    print(shlex.join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def _vllm_executable() -> str:
    """Resolve vLLM even when this venv's bin directory is absent from PATH."""
    discovered = shutil.which("vllm")
    if discovered:
        return discovered
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.is_file():
        return str(sibling)
    return "vllm"


def command_models(_args: argparse.Namespace) -> int:
    print(json.dumps({name: spec.to_dict() for name, spec in MODELS.items()}, indent=2))
    return 0


def command_download(args: argparse.Namespace) -> int:
    spec = get_model(args.model)
    destination = _model_path(args.model, args.destination)
    if args.dry_run:
        print(json.dumps({"source": spec.source, "destination": str(destination)}, indent=2))
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install the llama extra: pip install -e '.[llama]'", file=sys.stderr)
        return 2
    destination.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=spec.source,
            local_dir=destination,
            token=args.token or os.environ.get("HF_TOKEN"),
        )
    except Exception as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        return 1
    print(destination)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    spec = get_model(args.model)
    if spec.family != "llama":
        print(
            "The production serve launcher currently supports Llama through vLLM.",
            file=sys.stderr,
        )
        return 2
    path = _model_path(args.model, args.model_path)
    if not args.dry_run and not detect_hardware()["gpus"]:
        print(
            "No NVIDIA GPU was detected. Run 'kernellab doctor' and verify "
            "the driver/container runtime.",
            file=sys.stderr,
        )
        return 2
    executable = _vllm_executable()
    command = [
        executable,
        "serve",
        str(path),
        "--served-model-name",
        args.model,
        "--dtype",
        args.dtype or spec.default_dtype,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--chat-template",
        str(Path(args.chat_template).expanduser().resolve()),
    ]
    if args.disable_prefix_caching:
        command.append("--no-enable-prefix-caching")
    if args.quantization != "none":
        command.extend(["--quantization", args.quantization])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.enable_chunked_prefill:
        command.append("--enable-chunked-prefill")
    if args.attention_backend != "auto":
        command.extend(["--attention-backend", args.attention_backend])
    if args.max_num_seqs:
        command.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.tensor_parallel_size > 1:
        command.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    return _run(command, args.dry_run)


def command_benchmark(args: argparse.Namespace) -> int:
    if args.model != "llama-3.2-3b":
        print("The schema-v2 benchmark command currently supports llama-3.2-3b.", file=sys.stderr)
        return 2
    path = _model_path(args.model, args.model_path)
    output = str(Path(args.output).resolve())
    if args.engine == "pytorch":
        command = [
            sys.executable,
            str(LLAMA / "benchmarks" / "bench_torch_baseline.py"),
            "--model-dir",
            str(path),
            "--input-tokens",
            str(args.input_tokens),
            "--output-tokens",
            str(args.output_tokens),
            "--concurrency",
            str(args.concurrency),
            "--warmups",
            str(args.warmups),
            "--repetitions",
            str(args.repetitions),
            "--seed",
            str(args.seed),
            "--output",
            output,
        ]
    else:
        command = [
            sys.executable,
            str(LLAMA / "benchmarks" / "bench_serving.py"),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--tokenizer",
            str(path),
            "--input-tokens",
            str(args.input_tokens),
            "--output-tokens",
            str(args.output_tokens),
            "--concurrency",
            str(args.concurrency),
            "--warmups",
            str(args.warmups),
            "--repetitions",
            str(args.repetitions),
            "--seed",
            str(args.seed),
            "--dtype",
            args.dtype,
            "--quantization",
            args.quantization,
            "--cuda-graphs",
            args.cuda_graphs,
            "--attention-backend",
            args.attention_backend,
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--output",
            output,
        ]
    status = _run(command, args.dry_run)
    if status or args.dry_run:
        return status
    try:
        warnings = validate_result_file(output)
    except (OSError, json.JSONDecodeError, ResultValidationError) as error:
        print(f"Benchmark produced an invalid result: {error}", file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def command_plan(args: argparse.Namespace) -> int:
    matrix = build_matrix(
        concurrency=tuple(args.concurrency),
        input_lengths=tuple(args.input_lengths),
        output_lengths=tuple(args.output_lengths),
        dtypes=tuple(args.dtypes),
        quantizations=tuple(args.quantizations),
        graph_modes=tuple(args.graph_modes),
        repetitions=args.repetitions,
        warmups=args.warmups,
        seed=args.seed,
    )
    text = json.dumps(matrix, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text, end="")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    try:
        warnings = validate_result_file(args.path)
    except (OSError, json.JSONDecodeError, ResultValidationError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print("VALID")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernellab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="show the model registry")
    models.set_defaults(handler=command_models)

    doctor = subparsers.add_parser("doctor", help="show detected hardware")
    doctor.set_defaults(handler=lambda _args: (print_hardware() or 0))

    download = subparsers.add_parser("download", help="download model weights")
    download.add_argument("model", choices=sorted(MODELS))
    download.add_argument("--destination")
    download.add_argument("--token")
    download.add_argument("--dry-run", action="store_true")
    download.set_defaults(handler=command_download)

    serve = subparsers.add_parser("serve", help="launch a production vLLM server")
    serve.add_argument("model", choices=sorted(MODELS))
    serve.add_argument("--model-path")
    serve.add_argument("--dtype")
    serve.add_argument("--quantization", default="none")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-model-len", type=int, default=8192)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--max-num-seqs", type=int, default=0)
    serve.add_argument("--chat-template", default=str(DEFAULT_LLAMA_CHAT_TEMPLATE))
    serve.add_argument("--attention-backend", default="auto")
    serve.add_argument("--enable-chunked-prefill", action="store_true")
    serve.add_argument(
        "--disable-prefix-caching", action=argparse.BooleanOptionalAction, default=True
    )
    serve.add_argument("--enforce-eager", action="store_true")
    serve.add_argument("--dry-run", action="store_true")
    serve.set_defaults(handler=command_serve)

    benchmark = subparsers.add_parser("benchmark", help="run a schema-v2 benchmark")
    benchmark.add_argument("model", choices=sorted(MODELS))
    benchmark.add_argument("--engine", choices=("pytorch", "vllm"), default="pytorch")
    benchmark.add_argument("--model-path")
    benchmark.add_argument("--base-url", default="http://127.0.0.1:8000")
    benchmark.add_argument("--input-tokens", type=int, default=128)
    benchmark.add_argument("--output-tokens", type=int, default=128)
    benchmark.add_argument("--concurrency", type=int, default=1)
    benchmark.add_argument("--dtype", default="float16")
    benchmark.add_argument("--quantization", default="none")
    benchmark.add_argument("--cuda-graphs", choices=("enabled", "disabled"), default="enabled")
    benchmark.add_argument("--attention-backend", default="auto")
    benchmark.add_argument("--tensor-parallel-size", type=int, default=1)
    benchmark.add_argument("--warmups", type=int, default=5)
    benchmark.add_argument("--repetitions", type=int, default=100)
    benchmark.add_argument("--seed", type=int, default=1234)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--dry-run", action="store_true")
    benchmark.set_defaults(handler=command_benchmark)

    plan = subparsers.add_parser("plan", help="write the complete matched GPU benchmark matrix")
    plan.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    plan.add_argument("--input-lengths", type=int, nargs="+", default=[8, 128, 512, 2048, 8192])
    plan.add_argument("--output-lengths", type=int, nargs="+", default=[32, 128, 256])
    plan.add_argument("--dtypes", nargs="+", default=["float16"])
    plan.add_argument(
        "--quantizations", nargs="+", default=["none", "int8_per_channel_weight_only"]
    )
    plan.add_argument("--graph-modes", nargs="+", default=["enabled", "disabled"])
    plan.add_argument("--warmups", type=int, default=5)
    plan.add_argument("--repetitions", type=int, default=100)
    plan.add_argument("--seed", type=int, default=1234)
    plan.add_argument("--output")
    plan.set_defaults(handler=command_plan)

    validate = subparsers.add_parser("validate-result", help="validate a schema-v2 JSON result")
    validate.add_argument("path")
    validate.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
