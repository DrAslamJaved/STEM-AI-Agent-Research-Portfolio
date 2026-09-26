"""Compare fixed-width and disagreement-scaled split-conformal intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import POLICIES


def analyze(predictions: dict, checkpoints: dict, checkpoint_sha256: str) -> dict:
    """Validate normalized interval records before computing coverage summaries."""
    if predictions.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Prediction records do not match the checkpoint file digest")
    settings = checkpoints["settings"]
    if not settings.get("normalized_conformal"):
        raise ValueError("Checkpoints do not include normalized conformal intervals")
    policies = POLICIES if settings.get("include_diversity") else POLICIES[:2]
    folds = [f"fold_{i}" for i in settings["folds"]]
    seeds, budgets = settings["seeds"], settings["budgets"]
    audits = {row["fold"]: row for row in checkpoints["split_audits"]}
    expected = {(f, s, b, p) for f in folds for s in seeds for b in budgets for p in policies}
    ensemble = {}
    for row in checkpoints["checkpoints"]:
        if row["model"] != "ensemble":
            continue
        key = row["fold"], row["seed"], row["budget"], row["policy"]
        if key in ensemble:
            raise ValueError("Duplicate ensemble checkpoint")
        if any(name not in row for name in ("normalized_coverage_90", "normalized_interval_width_ev",
                                             "normalized_score_radius", "normalized_spread_floor_ev")):
            raise ValueError("Checkpoint lacks normalized interval metrics")
        ensemble[key] = row
    if set(ensemble) != expected or set(audits) != set(folds):
        raise ValueError("Incomplete checkpoint protocol")
    grouped, positions = defaultdict(list), set()
    for row in predictions["records"]:
        key = row["fold"], row["seed"], row["budget"], row["policy"]
        position = row["test_position"]
        if key not in expected or not isinstance(position, int) or not 0 <= position < audits[key[0]]["test_count"]:
            raise ValueError("Unexpected prediction key or test position")
        if (*key, position) in positions:
            raise ValueError("Duplicate prediction position")
        if (not isinstance(row.get("covered_90"), bool)
                or not isinstance(row.get("normalized_covered_90"), bool)
                or not np.isfinite(row.get("normalized_interval_width_ev", np.nan))
                or row["normalized_interval_width_ev"] < 0):
            raise ValueError("Invalid normalized prediction record")
        positions.add((*key, position))
        grouped[key].append(row)
    if set(grouped) != expected:
        raise ValueError("Missing prediction-level checkpoints")
    for key, rows in grouped.items():
        checkpoint = ensemble[key]
        if len(rows) != audits[key[0]]["test_count"]:
            raise ValueError("Missing test records")
        if (not np.isclose(np.mean([r["normalized_covered_90"] for r in rows]),
                          checkpoint["normalized_coverage_90"], atol=1e-10)
                or not np.isclose(np.mean([r["normalized_interval_width_ev"] for r in rows]),
                                  checkpoint["normalized_interval_width_ev"], atol=1e-10)):
            raise ValueError("Normalized prediction records disagree with checkpoint")
    summary = []
    for budget in budgets:
        for policy in policies:
            fields = {name: [] for name in ("fixed_coverage", "normalized_coverage", "fixed_width", "normalized_width",
                                             "fixed_low", "fixed_high", "normalized_low", "normalized_high")}
            for fold in folds:
                per_seed = {name: [] for name in fields}
                for seed in seeds:
                    rows = grouped[fold, seed, budget, policy]
                    spread = np.array([r["disagreement_ev"] for r in rows])
                    low, high = np.quantile(spread, [1 / 3, 2 / 3])
                    subsets = {"low": [r for r in rows if r["disagreement_ev"] <= low],
                               "high": [r for r in rows if r["disagreement_ev"] > high]}
                    values = {
                        "fixed_coverage": np.mean([r["covered_90"] for r in rows]),
                        "normalized_coverage": np.mean([r["normalized_covered_90"] for r in rows]),
                        "fixed_width": ensemble[fold, seed, budget, policy]["interval_width_ev"],
                        "normalized_width": np.mean([r["normalized_interval_width_ev"] for r in rows]),
                        "fixed_low": np.mean([r["covered_90"] for r in subsets["low"]]),
                        "fixed_high": np.mean([r["covered_90"] for r in subsets["high"]]),
                        "normalized_low": np.mean([r["normalized_covered_90"] for r in subsets["low"]]),
                        "normalized_high": np.mean([r["normalized_covered_90"] for r in subsets["high"]]),
                    }
                    for name, value in values.items():
                        per_seed[name].append(float(value))
                for name in fields:
                    fields[name].append(float(np.mean(per_seed[name])))
            summary.append({"budget": budget, "policy": policy,
                            **{name: float(np.mean(values)) for name, values in fields.items()}})
    return {"task": checkpoints["task"], "folds": len(folds), "seeds": len(seeds),
            "prediction_records": len(positions), "summary": summary,
            "floor_quantile": settings["normalized_spread_floor_quantile"]}


def render(report: dict) -> str:
    lines = ["# Project 09: Phase 06 normalized conformal intervals", "",
             f"Task: `{report['task']}` | folds: {report['folds']} | seeds: {report['seeds']} | "
             f"evaluation records: {report['prediction_records']}", "",
             f"Normalized intervals use finite-sample split-conformal scores of residual divided by "
             f"max(ensemble disagreement, calibration {report['floor_quantile']:.0%}-quantile floor). "
             "The calibration fold fixes the score quantile and floor before test evaluation.", "",
             "| Labels | Policy | Fixed coverage | Scaled coverage | Fixed width (eV) | Scaled width (eV) | "
             "Fixed high-spread coverage | Scaled high-spread coverage |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["summary"]:
        lines.append(f"| {row['budget']} | {row['policy']} | {row['fixed_coverage']:.3f} | "
                     f"{row['normalized_coverage']:.3f} | {row['fixed_width']:.3f} | "
                     f"{row['normalized_width']:.3f} | {row['fixed_high']:.3f} | "
                     f"{row['normalized_high']:.3f} |")
    lines += ["", "## Low-versus-high disagreement coverage", "",
              "| Labels | Policy | Fixed low | Fixed high | Scaled low | Scaled high |",
              "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in report["summary"]:
        lines.append(f"| {row['budget']} | {row['policy']} | {row['fixed_low']:.3f} | "
                     f"{row['fixed_high']:.3f} | {row['normalized_low']:.3f} | "
                     f"{row['normalized_high']:.3f} |")
    lines += ["", "## Interpretation", "",
              "The high-disagreement thirds are evaluated after training and are descriptive. Compare scaled "
              "coverage and width together: higher coverage obtained only by much wider intervals may not be useful. "
              "This phase studies uncertainty calibration; it does not alter acquisitions, models, budgets, "
              "or the locked Phase 03 label-efficiency conclusion.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.checkpoints.read_bytes()
    report = analyze(json.loads(args.predictions.read_text(encoding="utf-8")),
                     json.loads(data), hashlib.sha256(data).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    print(f"Saved normalized conformal report to {args.output}")


if __name__ == "__main__":
    main()
