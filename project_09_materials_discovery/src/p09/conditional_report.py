"""Audit prediction-level errors and descriptive conditional interval coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import POLICIES, _percentile_ranks


def _correlation(rows: list[dict]) -> float | None:
    spread = _percentile_ranks(np.array([r["disagreement_ev"] for r in rows]))
    error = _percentile_ranks(np.array([r["absolute_error_ev"] for r in rows]))
    if len(rows) < 3 or np.std(spread) == 0 or np.std(error) == 0:
        return None
    return float(np.corrcoef(spread, error)[0, 1])


def analyze(predictions: dict, checkpoints: dict, checkpoint_sha256: str) -> dict:
    """Check exact test-row grids and aggregate consistency before stratifying."""
    if predictions.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Prediction records do not match the checkpoint file digest")
    if (predictions.get("task"), predictions.get("split_mode")) != (
            checkpoints["task"], checkpoints.get("split_mode", "official")):
        raise ValueError("Prediction records have a different task or split")
    settings = checkpoints["settings"]
    policies = POLICIES if settings.get("include_diversity") else POLICIES[:2]
    folds = [f"fold_{i}" for i in settings["folds"]]
    seeds, budgets = settings["seeds"], settings["budgets"]
    audits = {a["fold"]: a for a in checkpoints["split_audits"]}
    if len(audits) != len(folds) or set(audits) != set(folds):
        raise ValueError("Incomplete split audits")
    ensemble = {(r["fold"], r["seed"], r["budget"], r["policy"]): r
                for r in checkpoints["checkpoints"] if r["model"] == "ensemble"}
    expected = {(f, s, b, p) for f in folds for s in seeds for b in budgets for p in policies}
    if set(ensemble) != expected or len(ensemble) != len([r for r in checkpoints["checkpoints"]
                                                          if r["model"] == "ensemble"]):
        raise ValueError("Incomplete or duplicate ensemble checkpoints")
    grouped = defaultdict(list)
    positions = set()
    for record in predictions["records"]:
        key = (record["fold"], record["seed"], record["budget"], record["policy"])
        position = record["test_position"]
        if key not in expected or not isinstance(position, int) or not 0 <= position < audits[key[0]]["test_count"]:
            raise ValueError("Unexpected prediction key or test position")
        if (*key, position) in positions:
            raise ValueError("Duplicate prediction position")
        positions.add((*key, position))
        if (not isinstance(record["covered_90"], bool) or record["element_count"] < 1
                or any(not np.isfinite(record[name]) or record[name] < 0
                       for name in ("absolute_error_ev", "disagreement_ev"))):
            raise ValueError("Invalid prediction-level evaluation field")
        grouped[key].append(record)
    if set(grouped) != expected:
        raise ValueError("Missing prediction-level checkpoints")
    for key, rows in grouped.items():
        checkpoint = ensemble[key]
        if len(rows) != audits[key[0]]["test_count"] or len(rows) != checkpoint["test_count"]:
            raise ValueError("Missing test records")
        if (not np.isclose(np.mean([r["absolute_error_ev"] for r in rows]), checkpoint["mae_ev"], atol=1e-10)
                or not np.isclose(np.mean([r["covered_90"] for r in rows]),
                                  checkpoint["coverage_90"], atol=1e-10)):
            raise ValueError("Per-material errors or coverage disagree with checkpoint")
    summaries = []
    for budget in budgets:
        for policy in policies:
            fold_means, fold_coverage, low_coverage, high_coverage, correlations = [], [], [], [], []
            for fold in folds:
                fold_means.append(float(np.mean([np.mean([r["absolute_error_ev"] for r in grouped[fold, s, budget, policy]])
                                                 for s in seeds])))
                fold_coverage.append(float(np.mean([np.mean([r["covered_90"] for r in grouped[fold, s, budget, policy]])
                                                    for s in seeds])))
                for seed in seeds:
                    rows = grouped[fold, seed, budget, policy]
                    values = np.array([r["disagreement_ev"] for r in rows])
                    lo, hi = np.quantile(values, [1/3, 2/3])
                    for subset, out in (([r for r in rows if r["disagreement_ev"] <= lo], low_coverage),
                                        ([r for r in rows if r["disagreement_ev"] > hi], high_coverage)):
                        if subset:
                            out.append(float(np.mean([r["covered_90"] for r in subset])))
                    rho = _correlation(rows)
                    if rho is not None:
                        correlations.append(rho)
            summaries.append({"budget": budget, "policy": policy,
                              "mae_ev": float(np.mean(fold_means)),
                              "coverage_90": float(np.mean(fold_coverage)),
                              "low_disagreement_coverage": float(np.mean(low_coverage)) if low_coverage else None,
                              "high_disagreement_coverage": float(np.mean(high_coverage)) if high_coverage else None,
                              "spearman_disagreement_error": float(np.mean(correlations)) if correlations else None})
    # Predeclared composition strata: one, two, or at least three elements.
    strata = []
    final_budget = budgets[-1]
    for policy in policies:
        for label, include in (("one", lambda n: n == 1), ("two", lambda n: n == 2),
                               ("three_plus", lambda n: n >= 3)):
            fold_coverage, fold_mae, minimum = [], [], None
            for fold in folds:
                seed_coverage, seed_mae = [], []
                for seed in seeds:
                    subset = [r for r in grouped[fold, seed, final_budget, policy] if include(r["element_count"])]
                    minimum = len(subset) if minimum is None else min(minimum, len(subset))
                    if subset:
                        seed_coverage.append(float(np.mean([r["covered_90"] for r in subset])))
                        seed_mae.append(float(np.mean([r["absolute_error_ev"] for r in subset])))
                if len(seed_coverage) == len(seeds):
                    fold_coverage.append(float(np.mean(seed_coverage)))
                    fold_mae.append(float(np.mean(seed_mae)))
            # Do not print unstable or empty subgroup estimates as meaningful numbers.
            sufficient = minimum is not None and minimum >= 20 and len(fold_coverage) == len(folds)
            strata.append({"policy": policy, "elements": label, "minimum_test_records_per_fold": minimum,
                           "mae_ev": float(np.mean(fold_mae)) if sufficient else None,
                           "coverage_90": float(np.mean(fold_coverage)) if sufficient else None})
    return {"task": checkpoints["task"], "split_mode": checkpoints.get("split_mode", "official"),
            "folds": len(folds), "seeds": len(seeds), "prediction_records": len(positions),
            "checkpoint_sha256": checkpoint_sha256,
            "budget_summary": summaries, "final_budget": final_budget, "element_strata": strata}


def render(report: dict) -> str:
    fmt = lambda value: "insufficient" if value is None else f"{value:.3f}"
    lines = ["# Project 09: Phase 05 prediction-level diagnostics", "",
             f"Task: `{report['task']}` | split: `{report['split_mode']}` | "
             f"folds: {report['folds']} | seeds: {report['seeds']} | "
             f"evaluation records: {report['prediction_records']}", "",
             "Each fold/seed/policy/budget evaluates the same fixed outer-test records. "
             "Coverage and MAE are descriptive fold means; seeds are averaged within folds. "
             "The low and high disagreement columns summarize test-set thirds defined only by predicted spread.", "",
             "| Labels | Policy | MAE (eV) | Coverage | Low spread coverage | High spread coverage | "
             "Spread-error Spearman |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["budget_summary"]:
        lines.append(f"| {row['budget']} | {row['policy']} | {fmt(row['mae_ev'])} | "
                     f"{fmt(row['coverage_90'])} | {fmt(row['low_disagreement_coverage'])} | "
                     f"{fmt(row['high_disagreement_coverage'])} | {fmt(row['spearman_disagreement_error'])} |")
    lines += ["", f"## Element-count strata at {report['final_budget']} labels", "",
              "Strata are fixed as one element, two elements, and three or more elements. "
              "A cell is withheld unless every fold/seed has at least 20 test records in that stratum.", "",
              "| Policy | Element stratum | Smallest fold/seed count | MAE (eV) | Coverage |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in report["element_strata"]:
        lines.append(f"| {row['policy']} | {row['elements']} | {row['minimum_test_records_per_fold']} | "
                     f"{fmt(row['mae_ev'])} | {fmt(row['coverage_90'])} |")
    lines += ["", "## Interpretation", "",
              "These subgroup analyses follow inspection of the Phase 03 results and are exploratory. "
              "Nominal 90% conformal coverage is a marginal statement under its assumptions; subgroup "
              "shortfalls do not by themselves identify their cause. Repeated test records across seeds "
              "are not independent observations. The locked Phase 03 ensemble conclusion remains unchanged. "
              "No test outcomes enter acquisition.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_bytes = args.checkpoints.read_bytes()
    result = analyze(json.loads(args.predictions.read_text(encoding="utf-8")),
                     json.loads(checkpoint_bytes), hashlib.sha256(checkpoint_bytes).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(result), encoding="utf-8")
    print(f"Saved diagnostic report to {args.output}")


if __name__ == "__main__":
    main()
