"""Report matched random, disagreement and disagreement-plus-diversity runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import POLICIES


def analyze(result: dict) -> dict:
    rows = [r for r in result["checkpoints"] if r["model"] == "ensemble"]
    paired = {}
    for row in rows:
        if row["policy"] not in POLICIES:
            raise ValueError(f"Unknown policy: {row['policy']}")
        key = (row["fold"], row["seed"], row["budget"], row["policy"])
        if key in paired:
            raise ValueError("Duplicate policy checkpoint")
        paired[key] = row
    triplets = {(fold, seed, budget) for fold, seed, budget, _ in paired}
    if not triplets or len(paired) != 3 * len(triplets) or any(
        (fold, seed, budget, policy) not in paired
        for fold, seed, budget in triplets for policy in POLICIES
    ):
        raise ValueError("Missing or unmatched three-policy checkpoints")
    folds = sorted({k[0] for k in triplets})
    seeds = sorted({k[1] for k in triplets})
    budgets = sorted({k[2] for k in triplets})
    if triplets != {(f, s, b) for f in folds for s in seeds for b in budgets}:
        raise ValueError("Incomplete fold/seed/budget grid")
    rng = np.random.default_rng(1909)
    summaries = []
    for budget in budgets:
        means = {p: [] for p in POLICIES}
        coverages = {p: [] for p in POLICIES}
        for fold in folds:
            for policy in POLICIES:
                matches = [paired[fold, seed, budget, policy] for seed in seeds]
                means[policy].append(float(np.mean([r["mae_ev"] for r in matches])))
                coverages[policy].append(float(np.mean([r["coverage_90"] for r in matches])))
        effects = {}
        for comparator in POLICIES[:2]:
            values = np.asarray(means[comparator]) - np.asarray(means["uncertainty_diversity"])
            ci = None
            if len(folds) >= 3:
                draws = rng.choice(values, size=(2000, len(folds)), replace=True).mean(axis=1)
                ci = [float(q) for q in np.quantile(draws, [0.025, 0.975])]
            effects[comparator] = {"gain_ev": float(np.mean(values)), "fold_bootstrap_ci95_ev": ci}
        summaries.append({"budget": budget,
                          "mae_ev": {p: float(np.mean(means[p])) for p in POLICIES},
                          "coverage_90": {p: float(np.mean(coverages[p])) for p in POLICIES},
                          "hybrid_gain_vs": effects})
    crossing_by_pair = {}
    for row in result["threshold_summary"]:
        key = (row["fold"], row["seed"])
        if key in crossing_by_pair:
            raise ValueError("Duplicate threshold crossing")
        first = row["first_budget"]
        if set(first) != set(POLICIES) or any(v is not None and v not in budgets for v in first.values()):
            raise ValueError("Invalid threshold crossing")
        crossing_by_pair[key] = first
    if set(crossing_by_pair) != {(f, s) for f in folds for s in seeds}:
        raise ValueError("Missing threshold crossing")
    counts = {p: sum(crossing_by_pair[f, s][p] is not None for f in folds for s in seeds)
              for p in POLICIES}
    paired_savings = {}
    for comparator in POLICIES[:2]:
        paired_savings[comparator] = [crossing_by_pair[f, s][comparator]
                                       - crossing_by_pair[f, s]["uncertainty_diversity"]
                                       for f in folds for s in seeds
                                       if crossing_by_pair[f, s][comparator] is not None
                                       and crossing_by_pair[f, s]["uncertainty_diversity"] is not None]
    return {"budget_summary": summaries, "crossing_counts": counts,
            "paired_savings_where_both_cross": paired_savings,
            "n_folds": len(folds), "n_seeds": len(seeds),
            "folds": folds, "seeds": seeds, "n_pairs": len(crossing_by_pair)}


def render(result: dict, summary: dict) -> str:
    lines = ["# Project 09: composition-diversity acquisition", "",
             f"Task: `{result['task']}` | split mode: `{result.get('split_mode', 'official')}` | "
             f"folds: {summary['n_folds']} | seeds: {summary['n_seeds']}", "",
             "The hybrid ranks bootstrap disagreement and distance to the closest labelled composition "
             "by percentile, then adds them with fixed equal weights. Distance uses the 118 element fractions. "
             "A positive gain means the hybrid has lower MAE at the same budget. "
             "Coverage is for nominal 90% split-conformal intervals.", "",
             "| Labels | Random MAE | Uncertainty MAE | Hybrid MAE | Hybrid gain vs random (95% CI) | "
             "Hybrid gain vs uncertainty (95% CI) | Random / uncertainty / hybrid coverage |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["budget_summary"]:
        def effect(policy):
            entry = row["hybrid_gain_vs"][policy]
            ci = entry["fold_bootstrap_ci95_ev"]
            return f"{entry['gain_ev']:+.3f} ({f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else 'insufficient folds'})"
        mae, cover = row["mae_ev"], row["coverage_90"]
        lines.append(f"| {row['budget']} | {mae['random']:.3f} | {mae['uncertainty']:.3f} | "
                     f"{mae['uncertainty_diversity']:.3f} | {effect('random')} | {effect('uncertainty')} | "
                     f"{cover['random']:.3f} / {cover['uncertainty']:.3f} / "
                     f"{cover['uncertainty_diversity']:.3f} |")
    lines += ["", f"## Threshold: {result['settings']['target_mae_ev']:.3f} eV MAE", "",
              f"Runs crossing at a scheduled budget (out of {summary['n_pairs']}): "
              + "; ".join(f"{p} {summary['crossing_counts'][p]}" for p in POLICIES) + ".", ""]
    for comparator, saved in summary["paired_savings_where_both_cross"].items():
        lines.append(f"Hybrid vs {comparator}: both crossed in {len(saved)} paired runs; "
                     + (f"median label saving {np.median(saved):g} (range {min(saved):g} to {max(saved):g})."
                        if saved else "label saving undefined.") )
    lines += ["", "## Interpretation", ""]
    if summary["n_folds"] < 5 or summary["n_seeds"] < 2:
        lines += ["Pilot only: finish five folds and two seeds before comparing policies scientifically.", ""]
    if result.get("split_mode", "official") == "group_exclusive":
        lines += ["This filtered training set is a robustness analysis, not an official Matbench score.", ""]
    lines += ["Uncrossed runs remain in the crossing counts; label savings include only pairs where both crossed. "
              "A five-fold bootstrap interval is descriptive and imprecise. Acquisition is retrospective: "
              "no new material was synthesized.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    summary = analyze(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(result, summary), encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
