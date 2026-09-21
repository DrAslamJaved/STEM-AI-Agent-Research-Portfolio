from __future__ import annotations

import json
from typing import Any

from causal_audit_agent.orchestrator import AnalysisRun


def _classification_to_dict(run: AnalysisRun) -> dict[str, Any] | None:
    if run.classification is None:
        return None
    return {
        "label": run.classification.label.value,
        "confidence": run.classification.confidence,
        "matched_cues": list(run.classification.matched_cues),
        "requires_human_review": run.classification.requires_human_review,
    }


def _serialize(component: object | None) -> dict[str, Any] | None:
    if component is None:
        return None
    to_dict = getattr(component, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"Component {type(component).__name__} is not serializable")
    return to_dict()


def analysis_run_to_dict(run: AnalysisRun) -> dict[str, Any]:
    """Return a stable, JSON-ready audit record for an analysis run."""

    return {
        "decision": {
            "state": run.state.value,
            "requires_human_review": run.requires_human_review,
            "numerical_causal_conclusion_emitted": (
                run.estimate is not None and run.estimate.estimate is not None
            ),
        },
        "question": run.question.to_dict(),
        "classification": _classification_to_dict(run),
        "graph_audit": _serialize(run.graph_audit),
        "identification": _serialize(run.identification),
        "estimation": _serialize(run.estimate),
        "advanced_estimation": _serialize(run.advanced_estimate),
        "diagnostics": _serialize(run.diagnostics),
        "refutation": _serialize(run.refutation),
        "sensitivity": _serialize(run.sensitivity),
        "benchmark": _serialize(run.benchmark_metrics),
        "audit_trail": [event.to_dict() for event in run.audit_trail],
        "warnings": list(run.warnings),
    }


def render_json(run: AnalysisRun, *, indent: int = 2) -> str:
    return json.dumps(
        analysis_run_to_dict(run),
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _section(lines: list[str], title: str) -> None:
    lines.extend(["", f"## {title}", ""])


def render_markdown(run: AnalysisRun) -> str:
    """Render a qualified report that separates every causal workflow stage."""

    lines = [
        "# Causal analysis and assumption-audit report",
        "",
        f"**Decision state:** `{run.state.value}`",
        "",
        "**Human approval required:** `yes`",
        "",
        "> This report does not convert observational evidence into proof of causation.",
    ]

    _section(lines, "Causal question")
    lines.extend(
        [
            f"- Question: {run.question.question}",
            f"- Treatment: `{run.question.treatment}`",
            f"- Outcome: `{run.question.outcome}`",
            f"- Requested estimand: `{run.question.estimand}`",
            f"- Population: {run.question.population}",
        ]
    )
    if run.classification is None:
        lines.append("- Classification: unavailable")
    else:
        lines.extend(
            [
                f"- Classification: `{run.classification.label.value}`",
                f"- Classification confidence: {run.classification.confidence:.3f}",
            ]
        )

    _section(lines, "DAG and assumption audit")
    if run.graph_audit is None:
        lines.append("DAG audit was not reached.")
    else:
        lines.extend(
            [
                "- Eligible for identification review: "
                f"`{str(run.graph_audit.valid_for_identification_review).lower()}`",
                f"- Errors: {list(run.graph_audit.errors)}",
                f"- Warnings: {list(run.graph_audit.warnings)}",
                f"- Missing assumptions: {list(run.graph_audit.missing_assumptions)}",
            ]
        )

    _section(lines, "Identification")
    if run.identification is None:
        lines.append("No estimand was identified.")
    else:
        lines.extend(
            [
                f"- Status: `{run.identification.status.value}`",
                f"- Method: `{run.identification.method}`",
                f"- Expression: `{run.identification.expression}`",
                f"- Adjustment sets: {list(run.identification.adjustment_sets)}",
                f"- Reasons: {list(run.identification.reasons)}",
            ]
        )

    _section(lines, "Estimation")
    if run.estimate is None or run.estimate.estimate is None:
        lines.append("No numerical causal estimate was emitted.")
    else:
        lines.extend(
            [
                f"- Status: `{run.estimate.status.value}`",
                f"- Method: `{run.estimate.method}`",
                f"- Estimate: {run.estimate.estimate:.6g}",
                f"- Standard error: {run.estimate.standard_error}",
                f"- Confidence interval: {run.estimate.confidence_interval}",
                f"- Adjustment set: {list(run.estimate.adjustment_set)}",
            ]
        )

    _section(lines, "Advanced estimation")
    if run.advanced_estimate is None:
        lines.append("Advanced estimation was not requested or was not reached.")
    else:
        lines.extend(
            [
                f"- Status: `{run.advanced_estimate.status.value}`",
                f"- Method: `{run.advanced_estimate.method}`",
                f"- Estimate: {run.advanced_estimate.estimate}",
                "- Unit-level effects available: "
                f"`{str(run.advanced_estimate.unit_effects is not None).lower()}`",
            ]
        )

    _section(lines, "Overlap, balance, and weight diagnostics")
    if run.diagnostics is None:
        lines.append("Diagnostics were not run.")
    else:
        lines.extend(
            [
                f"- Status: `{run.diagnostics.status.value}`",
                f"- Maximum weighted absolute SMD: {run.diagnostics.max_abs_smd:.6g}",
                f"- Overlap fraction: {run.diagnostics.overlap_fraction:.6g}",
                f"- Effective sample fraction: {run.diagnostics.effective_sample_fraction:.6g}",
                f"- Maximum weight: {run.diagnostics.max_weight:.6g}",
                f"- Reasons: {list(run.diagnostics.reasons)}",
            ]
        )

    _section(lines, "Refutation and stress testing")
    if run.refutation is None:
        lines.append("Refutation tests were not run.")
    else:
        lines.extend(
            [
                f"- Status: `{run.refutation.status.value}`",
                f"- Baseline estimate: {run.refutation.baseline_estimate}",
                f"- Checks: {[check.name for check in run.refutation.checks]}",
                f"- Reasons: {list(run.refutation.reasons)}",
            ]
        )

    _section(lines, "Hidden-confounding sensitivity")
    if run.sensitivity is None:
        lines.append("Sensitivity analysis was not run.")
    else:
        lines.extend(
            [
                f"- Status: `{run.sensitivity.status.value}`",
                f"- Analysis: `{run.sensitivity.analysis}`",
                f"- Robustness value: {run.sensitivity.robustness_value}",
                f"- E-value: {run.sensitivity.e_value}",
                f"- Reasons: {list(run.sensitivity.reasons)}",
            ]
        )

    _section(lines, "Benchmark evaluation")
    if run.benchmark_metrics is None:
        lines.append("No benchmark dataset was attached to this run.")
    else:
        lines.extend(
            [
                f"- Status: `{run.benchmark_metrics.status.value}`",
                f"- Benchmark: `{run.benchmark_metrics.benchmark}`",
                f"- ATE error: {run.benchmark_metrics.ate_error}",
                f"- PEHE: {run.benchmark_metrics.pehe}",
                f"- Confidence-interval covered: {run.benchmark_metrics.ci_covered}",
                f"- Reasons: {list(run.benchmark_metrics.reasons)}",
            ]
        )

    _section(lines, "Audit trail")
    lines.extend(
        f"- `{event.stage.value}` — `{event.status}`: {event.detail}"
        for event in run.audit_trail
    )

    _section(lines, "Warnings and interpretation gate")
    if run.warnings:
        lines.extend(f"- {warning}" for warning in run.warnings)
    else:
        lines.append("- No additional automated warning was emitted.")
    lines.extend(
        [
            "- Human approval of the DAG and final interpretation remains mandatory.",
            "- Refutation and sensitivity results measure fragility; they do not prove assumptions.",
        ]
    )
    return "\n".join(lines) + "\n"
