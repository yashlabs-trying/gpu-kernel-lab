#!/usr/bin/env python3
"""Compare a candidate NPZ capture with an FP16 reference capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from kernellab.quality import (  # noqa: E402
    compare_layers,
    compare_logits,
    negative_log_likelihood,
    release_gate,
    token_agreement,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.reference) as reference, np.load(args.candidate) as candidate:
        for required in ("logits", "generated_token_ids"):
            if required not in reference or required not in candidate:
                parser.error(f"both captures must contain {required!r}")
        logits = compare_logits(reference["logits"], candidate["logits"])
        layers = compare_layers(
            {key[6:]: reference[key] for key in reference.files if key.startswith("layer_")},
            {key[6:]: candidate[key] for key in candidate.files if key.startswith("layer_")},
        )
        tokens = token_agreement(
            reference["generated_token_ids"], candidate["generated_token_ids"]
        )
        reference_logits = reference["logits"].reshape(-1, reference["logits"].shape[-1])
        candidate_logits = candidate["logits"].reshape(-1, candidate["logits"].shape[-1])
        reference_tokens = reference["generated_token_ids"].reshape(-1)
        reference_nll = negative_log_likelihood(reference_logits, reference_tokens)
        candidate_nll = negative_log_likelihood(
            candidate_logits, reference_tokens
        )
        gate = release_gate(
            logits,
            token_metrics=tokens,
            reference_nll=reference_nll["nll"],
            candidate_nll=candidate_nll["nll"],
        )
        gate["checks"]["per_layer_cosine"] = (
            not layers["missing_layers"]
            and not layers["unexpected_layers"]
            and bool(layers["layers"])
            and all(metric["cosine_min"] >= 0.999 for metric in layers["layers"].values())
        )
        gate["passed"] = all(gate["checks"].values())
        result = {
            "schema_version": 1,
            "reference": str(args.reference),
            "candidate": str(args.candidate),
            "logits": logits,
            "tokens": tokens,
            "reference_nll": reference_nll,
            "candidate_nll_on_reference_tokens": candidate_nll,
            "per_layer": layers,
            "release_gate": gate,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
