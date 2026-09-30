"""Numerical quality metrics for reference and optimized LLM outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def _check_same_shape(reference: np.ndarray, candidate: np.ndarray) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    if reference.size == 0:
        raise ValueError("quality metrics require non-empty arrays")


def cosine_rows(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    _check_same_shape(reference, candidate)
    if reference.ndim < 2:
        reference = reference.reshape(1, -1)
        candidate = candidate.reshape(1, -1)
    else:
        reference = reference.reshape(-1, reference.shape[-1])
        candidate = candidate.reshape(-1, candidate.shape[-1])
    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    dot = np.sum(ref * cand, axis=-1)
    ref_norm = np.linalg.norm(ref, axis=-1)
    cand_norm = np.linalg.norm(cand, axis=-1)
    denominator = ref_norm * cand_norm
    result = np.zeros_like(dot)
    nonzero = denominator > 0
    result[nonzero] = dot[nonzero] / denominator[nonzero]
    result[(ref_norm == 0) & (cand_norm == 0)] = 1.0
    return np.clip(result, -1.0, 1.0)


def compare_logits(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    _check_same_shape(reference, candidate)
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    agreement = np.mean(np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1))
    cosines = cosine_rows(reference, candidate)
    return {
        "shape": list(reference.shape),
        "argmax_agreement": float(agreement),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_min": float(np.min(cosines)),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
    }


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, vocabulary]")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logit and label token counts differ")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("label is outside the vocabulary")
    stable = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(stable), axis=1))
    selected = stable[np.arange(labels.shape[0]), labels]
    nll = float(np.mean(logsumexp - selected))
    return {"nll": nll, "perplexity": float(math.exp(nll))}


def token_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference).reshape(-1)
    candidate = np.asarray(candidate).reshape(-1)
    compared = min(reference.size, candidate.size)
    matching = int(np.sum(reference[:compared] == candidate[:compared]))
    exact = reference.size == candidate.size and matching == reference.size
    return {
        "reference_tokens": int(reference.size),
        "candidate_tokens": int(candidate.size),
        "compared_tokens": int(compared),
        "matching_tokens": matching,
        "agreement": matching / compared if compared else 0.0,
        "exact_match": bool(exact),
    }


def compare_layers(
    reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    layers = {
        name: compare_logits(reference[name], candidate[name])
        for name in sorted(set(reference) & set(candidate))
    }
    return {"missing_layers": missing, "unexpected_layers": unexpected, "layers": layers}


def release_gate(
    logit_metrics: Mapping[str, Any],
    *,
    token_metrics: Mapping[str, Any] | None = None,
    reference_nll: float | None = None,
    candidate_nll: float | None = None,
    min_argmax: float = 0.99,
    min_cosine: float = 0.999,
    max_nll_regression: float = 0.01,
) -> dict[str, Any]:
    checks = {
        "argmax_agreement": float(logit_metrics["argmax_agreement"]) >= min_argmax,
        "minimum_cosine": float(logit_metrics["cosine_min"]) >= min_cosine,
    }
    if token_metrics is not None:
        checks["token_agreement"] = float(token_metrics["agreement"]) >= min_argmax
    if reference_nll is not None and candidate_nll is not None:
        checks["nll_regression"] = candidate_nll - reference_nll <= max_nll_regression
    return {"passed": all(checks.values()), "checks": checks}
