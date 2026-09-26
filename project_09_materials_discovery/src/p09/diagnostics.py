"""Audit Phase 03 result JSON and produce a model-aware evidence report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import POLICIES


MODELS = ("mean", "forest", "ensemble")
NUMERIC = ("mae_ev", "rmse_ev", "coverage_90", "interval_width_ev", "mean_disagreement_ev")


def validate(result: dict) -> dict:
    """Check the complete fold/seed/budget grid and saved selection invariants."""
    settings = result["settings"]
    if not settings.get("include_diversity"):
        raise ValueError("Phase 04 requires three-policy results")
    folds = [f"fold_{i}" for i in settings["folds"]]
    seeds, budgets = settings["seeds"], settings["budgets"]
    if not folds or not seeds or not budgets or len(set(folds)) != len(folds) or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid fold/seed grid")
    if sorted(set(budgets)) != budgets or budgets[0] != settings["initial"]:
        raise ValueError("Invalid budget schedule")
    audits = {a["fold"]: a for a in result["split_audits"]}
    if set(audits) != set(folds) or len(audits) != len(result["split_audits"]):
        raise ValueError("Missing or duplicate split audits")
    rows = {}
    for row in result["checkpoints"]:
        key = row["fold"], row["seed"], row["budget"], row["policy"], row["model"]
        if key in rows:
            raise ValueError("Duplicate checkpoint")
        rows[key] = row
    expected = {(fold, seed, budget, policy, model)
                for fold in folds for seed in seeds for budget in budgets
                for policy in POLICIES for model in MODELS}
    if set(rows) != expected:
        raise ValueError("Missing or unexpected checkpoints")
    for fold in folds:
        for seed in seeds:
            starts, calibrations = [], []
            for policy in POLICIES:
                previous = set()
                for budget in budgets:
                    ensemble = rows[fold, seed, budget, policy, "ensemble"]
                    chosen = ensemble["selected_train_positions"]
                    cal = ensemble["calibration_train_positions"]
                    chosen_set, cal_set = set(chosen), set(cal)
                    if (len(chosen) != budget or len(chosen_set) != budget
                            or len(cal) != len(cal_set)
                            or not previous.issubset(chosen_set) or chosen_set & cal_set
                            or any(not isinstance(i, int) or i < 0 or i >= audits[fold]["train_count"]
                                   for i in chosen_set | cal_set)):
                        raise ValueError("Invalid acquisition or calibration positions")
                    previous = chosen_set
                    calibrations.append(tuple(cal))
                starts.append(tuple(rows[fold, seed, budgets[0], policy, "ensemble"]
                                    ["selected_train_positions"]))
            if len(set(starts)) != 1 or len(set(calibrations)) != 1:
                raise ValueError("Policies have different initial labels or calibration sets")
    observed = {(row["fold"], row["seed"]): row for row in result["threshold_summary"]}
    if len(observed) != len(result["threshold_summary"]) or set(observed) != {
        (f, s) for f in folds for s in seeds
    }:
        raise ValueError("Missing or duplicate threshold summaries")
    target = settings["target_mae_ev"]
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Invalid target MAE")
    crossing = {}
    for model in ("forest", "ensemble"):
        crossing[model] = {}
        for fold in folds:
            for seed in seeds:
                crossing[model][fold, seed] = {}
                for policy in POLICIES:
                    eligible = [b for b in budgets
                                if rows[fold, seed, b, policy, model]["mae_ev"] <= target]
                    crossing[model][fold, seed][policy] = eligible[0] if eligible else None
                if model == "ensemble" and crossing[model][fold, seed] != observed[fold, seed]["first_budget"]:
                    raise ValueError("Stored ensemble crossing differs from checkpoints")
    return {"rows": rows, "audits": audits, "folds": folds, "seeds": seeds,
            "budgets": budgets, "crossing": crossing, "target": target}


def analyze(result: dict, robustness: dict | None = None) -> dict:
    protocol = validate(result)
    if result.get("split_mode", "official") != "official":
        raise ValueError("Primary result must use official folds")
    rows, folds, seeds, budgets = (protocol[k] for k in ("rows", "folds", "seeds", "budgets"))
    summary = []
    for budget in budgets:
        by_model = {}
        for model in MODELS:
            by_model[model] = {p: float(np.mean([
                rows[f, s, budget, p, model]["mae_ev"] for f in folds for s in seeds]))
                               for p in POLICIES}
        effects = {comparator: [float(np.mean([
            rows[f, s, budget, comparator, "ensemble"]["mae_ev"]
            - rows[f, s, budget, "uncertainty_diversity", "ensemble"]["mae_ev"]
            for s in seeds])) for f in folds] for comparator in POLICIES[:2]}
        summary.append({"budget": budget, "mae_ev": by_model,
                        "hybrid_gain_ev_by_fold": effects})
    crossings = {model: {policy: sum(protocol["crossing"][model][f, s][policy] is not None
                                      for f in folds for s in seeds)
                         for policy in POLICIES} for model in ("forest", "ensemble")}
    robustness_status = "not supplied"
    if robustness is not None:
        other = validate(robustness)
        if robustness.get("split_mode") != "group_exclusive":
            raise ValueError("Second result must be group_exclusive")
        if (result["task"] != robustness["task"] or result["settings"] != robustness["settings"]
                or result.get("versions") != robustness.get("versions")
                or (folds, seeds, budgets) != (other["folds"], other["seeds"], other["budgets"])):
            raise ValueError("Unmatched robustness protocol")
        for f in folds:
            a, b = protocol["audits"][f], other["audits"][f]
            if a["test_count"] != b["test_count"] or a["train_count"] - b["train_count"] != b["removed_training_records"]:
                raise ValueError("Inconsistent group-exclusive fold counts")
        removed = sum(other["audits"][f]["removed_training_records"] for f in folds)
        if removed == 0:
            for key in rows:
                first, second = rows[key], other["rows"][key]
                if any(not np.isclose(first[k], second[k], atol=1e-10, rtol=0)
                       for k in NUMERIC if k in first and k in second):
                    raise ValueError("Unfiltered robustness run has different metrics")
                if first.get("selected_train_positions") != second.get("selected_train_positions"):
                    raise ValueError("Unfiltered robustness run has different selections")
            robustness_status = "No training rows removed; results agree within 1e-10 eV. Not an independent robustness test."
        else:
            robustness_status = f"{removed} training rows removed across folds; analyze robustness report separately."
    return {"task": result["task"], "target_mae_ev": protocol["target"],
            "n_folds": len(folds), "n_seeds": len(seeds), "budgets": summary,
            "crossing_counts": crossings, "robustness": robustness_status,
            "outer_overlap_compositions": sum(a["original_overlapping_train_test_compositions"]
                                              for a in protocol["audits"].values())}


def render(summary: dict) -> str:
    lines = ["# Project 09: Phase 04 evidence audit", "",
             f"Task: `{summary['task']}` | official folds: {summary['n_folds']} | seeds: {summary['n_seeds']}",
             f"Locked ensemble MAE target: {summary['target_mae_ev']:.3f} eV", "",
             "| Labels | Random ensemble | Uncertainty ensemble | Hybrid ensemble | Random forest | Uncertainty forest | Hybrid forest |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for b in summary["budgets"]:
        ens, forest = b["mae_ev"]["ensemble"], b["mae_ev"]["forest"]
        lines.append(f"| {b['budget']} | {ens['random']:.3f} | {ens['uncertainty']:.3f} | "
                     f"{ens['uncertainty_diversity']:.3f} | {forest['random']:.3f} | "
                     f"{forest['uncertainty']:.3f} | {forest['uncertainty_diversity']:.3f} |")
    lines += ["", "## First crossing at the scheduled budgets", "",
              "| Model | Random | Uncertainty | Hybrid |", "| --- | ---: | ---: | ---: |"]
    for model in ("ensemble", "forest"):
        c = summary["crossing_counts"][model]
        lines.append(f"| {model} | {c['random']} | {c['uncertainty']} | {c['uncertainty_diversity']} |")
    lines += ["", f"Counts are out of {summary['n_folds'] * summary['n_seeds']} fold/seed runs. "
              "Forest crossings are exploratory; the ensemble threshold analysis remains primary.", "",
              "## Hybrid MAE gain by fold", "",
              "Positive means lower hybrid ensemble MAE; values average the predeclared seeds within a fold.", "",
              "| Labels | Hybrid vs random (folds) | Hybrid vs uncertainty (folds) |",
              "| ---: | --- | --- |"]
    for b in summary["budgets"]:
        effects = b["hybrid_gain_ev_by_fold"]
        values = lambda p: ", ".join(f"{v:+.3f}" for v in effects[p])
        lines.append(f"| {b['budget']} | {values('random')} | {values('uncertainty')} |")
    lines += ["", "## Split audit", "",
              f"Shared reduced compositions across official train/test folds (sum): {summary['outer_overlap_compositions']}.",
              summary["robustness"], "",
              "## Interpretation", "",
              "The model-specific forest crossings were inspected after viewing Phase 03 outcomes. "
              "Do not rebrand them as a predeclared success. This checkpoint-level record cannot diagnose "
              "per-material errors or conditional interval coverage; those require new prediction-level exports. "
              "All label acquisition here is retrospective.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Phase 03 official JSON")
    parser.add_argument("--robustness", type=Path, help="Phase 03 group-exclusive JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    primary = json.loads(args.input.read_text(encoding="utf-8"))
    other = json.loads(args.robustness.read_text(encoding="utf-8")) if args.robustness else None
    summary = analyze(primary, other)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary), encoding="utf-8")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved evidence audit to {args.output}")


if __name__ == "__main__":
    main()
