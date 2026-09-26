"""Write a conservative, fold-clustered comparison from experiment JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(result: dict) -> dict:
    rows = [r for r in result["checkpoints"] if r["model"] == "ensemble"]
    key = lambda r: (r["fold"], r["seed"], r["budget"])
    by_policy = {policy: {key(r): r for r in rows if r["policy"] == policy}
                 for policy in ("random", "uncertainty")}
    if not by_policy["random"] or by_policy["random"].keys() != by_policy["uncertainty"].keys():
        raise ValueError("Missing or unmatched paired policy checkpoints")
    if len(rows) != sum(map(len, by_policy.values())):
        raise ValueError("Duplicate paired checkpoint")
    rng = np.random.default_rng(1909)
    budgets = sorted({k[2] for k in by_policy["random"]})
    summaries = []
    for budget in budgets:
        fold_ids = sorted({k[0] for k in by_policy["random"] if k[2] == budget})
        effects, random_mae, active_mae, random_coverage, active_coverage = [], [], [], [], []
        for fold in fold_ids:
            pairs = [k for k in by_policy["random"] if k[0] == fold and k[2] == budget]
            if not pairs:
                raise ValueError("Missing fold at a budget")
            random_mae.append(float(np.mean([by_policy["random"][k]["mae_ev"] for k in pairs])))
            active_mae.append(float(np.mean([by_policy["uncertainty"][k]["mae_ev"] for k in pairs])))
            random_coverage.append(float(np.mean([by_policy["random"][k]["coverage_90"] for k in pairs])))
            active_coverage.append(float(np.mean([by_policy["uncertainty"][k]["coverage_90"] for k in pairs])))
            effects.append(random_mae[-1] - active_mae[-1])
        # Repeated seeds are averaged inside each fold; folds are resampling units.
        arr = np.asarray(effects)
        ci = None
        if len(arr) >= 3:
            draws = rng.choice(arr, size=(2000, len(arr)), replace=True).mean(axis=1)
            ci = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
        summaries.append({
            "budget": budget, "folds": len(fold_ids),
            "random_mae_ev": float(np.mean(random_mae)),
            "uncertainty_mae_ev": float(np.mean(active_mae)),
            "random_coverage_90": float(np.mean(random_coverage)),
            "uncertainty_coverage_90": float(np.mean(active_coverage)),
            "paired_mae_gain_ev": float(np.mean(arr)),
            "fold_bootstrap_ci95_ev": ci,
        })
    crossings = result["threshold_summary"]
    counts = {"both": 0, "uncertainty_only": 0, "random_only": 0, "neither": 0}
    saved = []
    for crossing in crossings:
        first = crossing["first_budget"]
        a, r = first["uncertainty"], first["random"]
        category = "both" if a is not None and r is not None else (
            "uncertainty_only" if a is not None else "random_only" if r is not None else "neither")
        counts[category] += 1
        if category == "both":
            saved.append(r - a)
    return {"budget_summary": summaries, "crossing_categories": counts,
            "labels_saved_where_both_cross": saved,
            "n_folds": len({r["fold"] for r in rows}),
            "n_seeds": len({r["seed"] for r in rows})}


def render(result: dict, summary: dict) -> str:
    mode = result.get("split_mode", "official")
    official = mode == "official"
    lines = ["# Project 09 learning-curve report", "",
             f"Protocol: `{mode}` | task: `{result['task']}` | unit: eV | "
             f"folds: {summary['n_folds']} | seeds: {summary['n_seeds']}", "",
             "The positive MAE gain is random MAE minus uncertainty MAE at the same label budget.", "",
             "| Labels | Random MAE | Uncertainty MAE | Gain | 95% fold-bootstrap CI | Random coverage | Uncertainty coverage |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["budget_summary"]:
        ci = r["fold_bootstrap_ci95_ev"]
        interval = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "insufficient folds"
        lines.append(f"| {r['budget']} | {r['random_mae_ev']:.3f} | {r['uncertainty_mae_ev']:.3f} | "
                     f"{r['paired_mae_gain_ev']:+.3f} | {interval} | "
                     f"{r['random_coverage_90']:.3f} | {r['uncertainty_coverage_90']:.3f} |")
    c = summary["crossing_categories"]
    lines += ["", f"## First crossing of {result['settings']['target_mae_ev']:.3f} eV MAE", "",
              f"Paired fold/seed runs: both {c['both']}; only uncertainty {c['uncertainty_only']}; "
              f"only random {c['random_only']}; neither {c['neither']}.", ""]
    saved = summary["labels_saved_where_both_cross"]
    if saved:
        lines += [f"Among runs where both crossed, median labels saved: {np.median(saved):g}; "
                  f"range: {min(saved):g} to {max(saved):g}. Negative means uncertainty required more labels.", ""]
    audits = result["split_audits"]
    lines += ["## Split audit", "", "| Fold | Training | Test | Shared compositions before filtering | Test records affected | Training rows removed |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for a in audits:
        shared = a.get("original_overlapping_train_test_compositions",
                       a.get("overlapping_train_test_compositions", 0))
        affected = a.get("original_overlapping_test_records",
                         a.get("overlapping_test_records", 0))
        removed = a.get("removed_training_records", 0)
        lines.append(f"| {a['fold']} | {a['train_count']} | {a['test_count']} | "
                     f"{shared} | {affected} | {removed} |")
    lines += ["", "## Interpretation rules", ""]
    if summary["n_folds"] < 5 or summary["n_seeds"] < 2:
        lines += ["This is a pilot: complete five official folds and at least two seeds before a scientific comparison.", ""]
    if official and any(a.get("original_overlapping_train_test_compositions",
                              a.get("overlapping_train_test_compositions", 0)) for a in audits):
        lines += ["Official train/test folds share some reduced compositions. Re-run with "
                  "`--split-mode group_exclusive` and report that robustness analysis separately.", ""]
    if not official:
        lines += ["Group-exclusive filtering changes the official training folds and is a robustness analysis, "
                  "not an official Matbench score. The outer test rows remain the same.", ""]
    lines += ["A fold-bootstrap interval with five folds is descriptive and imprecise. "
              "Conformal coverage is marginal under exchangeability; assess shortfall from 0.90 and conditional failures.",
              "This is retrospective acquisition among existing labelled records; no new material has been synthesized.", ""]
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
