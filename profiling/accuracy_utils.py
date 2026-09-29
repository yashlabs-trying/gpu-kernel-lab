"""CPU-testable utilities shared by Qwen numerical acceptance tools."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


PROMPT_TOPICS = (
    "GPU memory coalescing", "matrix multiplication", "continuous batching",
    "the water cycle", "renewable energy", "a railway journey", "binary search",
    "distributed systems", "probability", "healthy sleep", "photosynthesis",
    "database indexes", "the French Revolution", "network congestion",
    "a robot learning to paint", "attention in language models",
)
PROMPT_TASKS = (
    "Explain {topic} to a beginner in three precise sentences.",
    "Give a technically accurate summary of {topic}, then list two limitations.",
    "Compare {topic} with a closely related idea and provide one concrete example.",
    "Write a short question and answer about {topic}.",
    "Describe a practical debugging procedure involving {topic}.",
    "State the central idea of {topic}, a common mistake, and how to verify it.",
    "Teach {topic} using a small numerical example where appropriate.",
    "Write a concise paragraph about {topic} for a university study guide.",
)
PROMPT_SUFFIXES = (
    "Use plain language.", "Be concise but do not omit assumptions.",
    "Separate facts from inference.", "Include the important trade-off.",
    "End with a one-sentence conclusion.", "Avoid unnecessary jargon.",
)


def deterministic_prompts(count: int, seed: int = 20260904) -> list[str]:
    """Build a reproducible, copyright-free prompt set with varied subject matter."""
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    combinations = [
        f"{task.format(topic=topic)} {suffix}"
        for topic in PROMPT_TOPICS
        for task in PROMPT_TASKS
        for suffix in PROMPT_SUFFIXES
    ]
    rng.shuffle(combinations)
    if count <= len(combinations):
        return combinations[:count]
    prompts = []
    for index in range(count):
        base = combinations[index % len(combinations)]
        prompts.append(f"Example set {index // len(combinations) + 1}. {base}")
    return prompts


@dataclass
class StreamingTensorMetrics:
    count: int = 0
    max_abs: float = 0.0
    mean_abs_sum: float = 0.0
    min_cosine: float = 1.0
    finite: bool = True

    def update(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        if actual.shape != expected.shape:
            raise ValueError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
        a, e = actual.detach().float(), expected.detach().float()
        self.finite = self.finite and bool(torch.isfinite(a).all() and torch.isfinite(e).all())
        delta = (a - e).abs()
        self.max_abs = max(self.max_abs, float(delta.max()) if delta.numel() else 0.0)
        self.mean_abs_sum += float(delta.mean()) if delta.numel() else 0.0
        if a.numel():
            cosine = float(F.cosine_similarity(a.flatten(), e.flatten(), dim=0, eps=1e-12))
            self.min_cosine = min(self.min_cosine, cosine)
        self.count += 1

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs_sum / self.count if self.count else 0.0,
            "min_cosine": self.min_cosine,
            "finite": self.finite,
        }


def evaluate_gate(
    *, prefill_agreement: float, decode_agreement: float,
    minimum_layer_cosine: float, nll_delta: float, finite: bool,
    missing_checkpoints: int = 0, agreement_requirement: float = 0.99,
    cosine_requirement: float = 0.999, max_abs_nll_delta: float = 0.01,
) -> dict:
    checks = {
        "prefill_argmax": prefill_agreement >= agreement_requirement,
        "decode_argmax": decode_agreement >= agreement_requirement,
        "layer_cosine": minimum_layer_cosine >= cosine_requirement,
        "nll_delta": math.isfinite(nll_delta) and abs(nll_delta) <= max_abs_nll_delta,
        "finite": finite,
        "checkpoint_alignment": missing_checkpoints == 0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "requirements": {
            "minimum_prefill_argmax_agreement": agreement_requirement,
            "minimum_decode_argmax_agreement": agreement_requirement,
            "minimum_layer_cosine": cosine_requirement,
            "maximum_absolute_mean_nll_delta": max_abs_nll_delta,
            "all_values_finite": True,
            "missing_checkpoints": 0,
        },
    }
