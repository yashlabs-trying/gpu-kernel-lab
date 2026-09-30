"""Canonical model registry shared by CLI and benchmark tooling."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str
    family: str
    default_dtype: str
    local_directory: str
    gated: bool
    implementation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODELS = {
    "qwen3-0.6b": ModelSpec(
        name="qwen3-0.6b",
        source="Qwen/Qwen3-0.6B",
        family="qwen3",
        default_dtype="bfloat16",
        local_directory="qwen3-0.6b",
        gated=False,
        implementation="native-kernellab",
    ),
    "llama-3.2-3b": ModelSpec(
        name="llama-3.2-3b",
        source="meta-llama/Llama-3.2-3B",
        family="llama",
        default_dtype="float16",
        local_directory="llama-3.2-3b-hf",
        gated=True,
        implementation="vllm-and-transformers",
    ),
}


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as error:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model {name!r}; choose one of: {choices}") from error
