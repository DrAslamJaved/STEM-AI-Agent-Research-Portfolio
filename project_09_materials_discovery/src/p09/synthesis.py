"""Create an auditable Phase 09 synthesis from validated Project 09 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .conditional_report import analyze as analyze_phase05
from .diagnostics import analyze as analyze_phase03
from .normalized_report import analyze as analyze_phase06


PHASE08_MODELS = ("descriptor_random_forest", "periodic_message_passing")


def _digest(path: Path) -> dict[str, object]:
    """Return a stable, path-independent evidence record."""
    data = path.read_bytes()
    return {"filename": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _finite_nonnegative(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}")
    return float(value)


def summarize_phase08(result: dict) -> dict:
    """Validate and summarize the fixed descriptor-versus-graph comparison."""
    if (result.get("phase"), result.get("task"), result.get("benchmark")) != (
        "08", "matbench_dielectric", "Matbench v0.1"
    ):
        raise ValueError("Phase 08 artifact has an unexpected identity")
    protocol = result.get("protocol", {})
    if protocol.get("seed") != 17 or protocol.get("graph_model", {}).get("cpu_threads") != 1:
        raise ValueError("Phase 08 deterministic protocol is not locked")
    runs = result.get("runs")
    if not isinstance(runs, list) or len(runs) != 5:
        raise ValueError("Phase 08 requires exactly five official-fold runs")
    by_fold = {row.get("fold"): row for row in runs}
    if set(by_fold) != {0, 1, 2, 3, 4} or len(by_fold) != len(runs):
        raise ValueError("Phase 08 folds must be the unique official folds 0--4")
    summaries: dict[str, dict[str, float]] = {}
    for model in PHASE08_MODELS:
        metrics = []
        for fold in range(5):
            row = by_fold[fold]
            if row.get("seed") != 17 or not isinstance(row.get("train_count"), int) or row["train_count"] <= 0:
                raise ValueError("Invalid Phase 08 run metadata")
            values = row.get("models", {}).get(model)
            if not isinstance(values, dict) or not isinstance(values.get("test_count"), int) or values["test_count"] <= 0:
                raise ValueError(f"Missing {model} test metrics")
            metrics.append({
                "mae": _finite_nonnegative(values.get("mae"), f"{model} MAE"),
                "rmse": _finite_nonnegative(values.get("rmse"), f"{model} RMSE"),
                "test_count": values["test_count"],
            })
        count = sum(row["test_count"] for row in metrics)
        summaries[model] = {
            "mean_fold_mae": float(np.mean([row["mae"] for row in metrics])),
            "mean_fold_rmse": float(np.mean([row["rmse"] for row in metrics])),
            "weighted_mae": float(sum(row["mae"] * row["test_count"] for row in metrics) / count),
            "pooled_rmse": float(math.sqrt(sum(row["rmse"] ** 2 * row["test_count"] for row in metrics) / count)),
            "test_records": count,
        }
    rf, graph = summaries[PHASE08_MODELS[0]], summaries[PHASE08_MODELS[1]]
    return {
        "task": result["task"],
        "target": result.get("target"),
        "unit": result.get("unit"),
        "folds": 5,
        "models": summaries,
        "graph_minus_descriptor": {
            "mean_fold_mae": graph["mean_fold_mae"] - rf["mean_fold_mae"],
            "mean_fold_rmse": graph["mean_fold_rmse"] - rf["mean_fold_rmse"],
        },
        "conclusion": (
            "graph model did not outperform the descriptor baseline"
            if graph["mean_fold_mae"] >= rf["mean_fold_mae"]
            else "graph model outperformed the descriptor baseline"
        ),
    }


def _phase03_release_summary(result: dict) -> dict:
    report = analyze_phase03(result)
    crossing = report["crossing_counts"]["ensemble"]
    savings = [row.get("labels_saved_vs_random") for row in result["threshold_summary"]]
    hybrid_savings = [row.get("labels_saved_diversity_vs_random") for row in result["threshold_summary"]]
    positive = sum(value is not None and value > 0 for value in savings)
    hybrid_positive = sum(value is not None and value > 0 for value in hybrid_savings)
    return {
        "task": report["task"],
        "target_mae_ev": report["target_mae_ev"],
        "folds": report["n_folds"],
        "seeds": report["n_seeds"],
        "scheduled_budgets": [row["budget"] for row in report["budgets"]],
        "ensemble_crossings": crossing,
        "positive_uncertainty_savings": positive,
        "positive_hybrid_savings": hybrid_positive,
        "conclusion": (
            "no observed label savings at the locked ensemble target"
            if positive == 0 and hybrid_positive == 0
            else "at least one positive paired label-saving observation"
        ),
    }


def _phase05_release_summary(predictions: dict, checkpoints: dict, digest: str) -> dict:
    report = analyze_phase05(predictions, checkpoints, digest)
    correlations = [row["spearman_disagreement_error"] for row in report["budget_summary"]
                    if row["spearman_disagreement_error"] is not None]
    high_coverage = [row["high_disagreement_coverage"] for row in report["budget_summary"]
                     if row["high_disagreement_coverage"] is not None]
    return {
        "task": report["task"], "prediction_records": report["prediction_records"],
        "correlation_range": [float(min(correlations)), float(max(correlations))] if correlations else None,
        "high_disagreement_coverage_range": [float(min(high_coverage)), float(max(high_coverage))]
        if high_coverage else None,
    }


def _phase06_release_summary(predictions: dict, checkpoints: dict, digest: str) -> dict:
    report = analyze_phase06(predictions, checkpoints, digest)
    high_improvements = [row["normalized_high"] - row["fixed_high"] for row in report["summary"]]
    width_changes = [row["normalized_width"] - row["fixed_width"] for row in report["summary"]]
    return {
        "task": report["task"], "prediction_records": report["prediction_records"],
        "high_disagreement_coverage_change_range": [float(min(high_improvements)), float(max(high_improvements))],
        "interval_width_change_range_ev": [float(min(width_changes)), float(max(width_changes))],
    }


def synthesize(
    phase03_path: Path,
    phase08_path: Path,
    phase05_checkpoints: Path | None = None,
    phase05_predictions: Path | None = None,
    phase06_checkpoints: Path | None = None,
    phase06_predictions: Path | None = None,
) -> dict:
    """Validate supplied evidence and return a deterministic release record."""
    if (phase05_checkpoints is None) != (phase05_predictions is None):
        raise ValueError("Phase 05 requires both checkpoints and prediction records")
    if (phase06_checkpoints is None) != (phase06_predictions is None):
        raise ValueError("Phase 06 requires both checkpoints and prediction records")
    files = {"phase03": _digest(phase03_path), "phase08": _digest(phase08_path)}
    phase03 = json.loads(phase03_path.read_text(encoding="utf-8"))
    phase08 = json.loads(phase08_path.read_text(encoding="utf-8"))
    output = {"phase": "09", "evidence_files": files,
              "phase03": _phase03_release_summary(phase03), "phase08": summarize_phase08(phase08)}
    if phase05_checkpoints is not None and phase05_predictions is not None:
        checkpoint_bytes = phase05_checkpoints.read_bytes()
        files["phase05_checkpoints"] = _digest(phase05_checkpoints)
        files["phase05_predictions"] = _digest(phase05_predictions)
        output["phase05"] = _phase05_release_summary(
            json.loads(phase05_predictions.read_text(encoding="utf-8")),
            json.loads(checkpoint_bytes), hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
    if phase06_checkpoints is not None and phase06_predictions is not None:
        checkpoint_bytes = phase06_checkpoints.read_bytes()
        files["phase06_checkpoints"] = _digest(phase06_checkpoints)
        files["phase06_predictions"] = _digest(phase06_predictions)
        output["phase06"] = _phase06_release_summary(
            json.loads(phase06_predictions.read_text(encoding="utf-8")),
            json.loads(checkpoint_bytes), hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
    return output


def render(summary: dict) -> str:
    """Render a concise, claim-limited Markdown release report."""
    p03, p08 = summary["phase03"], summary["phase08"]
    rf = p08["models"]["descriptor_random_forest"]
    graph = p08["models"]["periodic_message_passing"]
    lines = ["# Project 09: Phase 09 evidence synthesis", "",
             "## Evidence identity", "", "| Phase | File | SHA-256 | Bytes |", "| --- | --- | --- | ---: |"]
    for label, record in summary["evidence_files"].items():
        lines.append(f"| {label} | `{record['filename']}` | `{record['sha256']}` | {record['bytes']} |")
    lines += ["", "## Predeclared active-learning conclusion", "",
              f"The official `{p03['task']}` analysis used {p03['folds']} folds, {p03['seeds']} seeds, "
              f"and the fixed {p03['target_mae_ev']:.2f} eV ensemble MAE target at "
              f"{p03['scheduled_budgets']} labelled records.", "",
              f"**Conclusion:** {p03['conclusion']}. Positive paired savings occurred in "
              f"{p03['positive_uncertainty_savings']} uncertainty and {p03['positive_hybrid_savings']} "
              "hybrid comparisons. Failed crossings remain failures, not zero savings.", "",
              "## Structure-model comparison", "",
              f"The deterministic `{p08['task']}` comparison used {p08['folds']} official folds and "
              f"{rf['test_records']} held-out records.", "",
              "| Model | Mean fold MAE | Mean fold RMSE |", "| --- | ---: | ---: |",
              f"| Descriptor random forest | {rf['mean_fold_mae']:.4f} | {rf['mean_fold_rmse']:.4f} |",
              f"| Periodic message-passing GNN | {graph['mean_fold_mae']:.4f} | {graph['mean_fold_rmse']:.4f} |", "",
              f"**Conclusion:** {p08['conclusion']} (GNN minus descriptor: "
              f"{p08['graph_minus_descriptor']['mean_fold_mae']:+.4f} MAE, "
              f"{p08['graph_minus_descriptor']['mean_fold_rmse']:+.4f} RMSE)."]
    if "phase05" in summary:
        p05 = summary["phase05"]
        lines += ["", "## Exploratory uncertainty diagnostics", "",
                  f"Phase 05 validated {p05['prediction_records']} prediction-level evaluation records. "
                  f"The descriptive disagreement-error correlation range was {p05['correlation_range']}; "
                  f"high-disagreement coverage range was {p05['high_disagreement_coverage_range']}."]
    if "phase06" in summary:
        p06 = summary["phase06"]
        lines += ["", f"Phase 06 validated {p06['prediction_records']} prediction-level evaluation records. "
                  f"Its scaled-minus-fixed high-disagreement coverage range was "
                  f"{p06['high_disagreement_coverage_change_range']}; interval-width change range (eV) was "
                  f"{p06['interval_width_change_range_ev']}."]
    lines += ["", "## Claim boundary", "",
              "This is retrospective benchmark evidence, not laboratory discovery. Phase 05/06 diagnostics "
              "are exploratory and do not change the locked Phase 03 active-learning conclusion. The Phase 08 "
              "graph comparison does not justify graph-based uncertainty sampling because its fixed model did not "
              "outperform the descriptor baseline. Future work should preregister a new model-selection and "
              "acquisition protocol before making any positive graph-assisted discovery claim.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase03", type=Path, required=True, help="Phase 03 official checkpoint JSON")
    parser.add_argument("--phase08", type=Path, required=True, help="Phase 08 deterministic official JSON")
    parser.add_argument("--phase05-checkpoints", type=Path)
    parser.add_argument("--phase05-predictions", type=Path)
    parser.add_argument("--phase06-checkpoints", type=Path)
    parser.add_argument("--phase06-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Markdown synthesis output")
    parser.add_argument("--manifest", type=Path, required=True, help="Machine-readable synthesis manifest")
    args = parser.parse_args()
    summary = synthesize(args.phase03, args.phase08, args.phase05_checkpoints, args.phase05_predictions,
                         args.phase06_checkpoints, args.phase06_predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary), encoding="utf-8")
    args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved evidence synthesis to {args.output}")
    print(f"Saved evidence manifest to {args.manifest}")


if __name__ == "__main__":
    main()
