"""Release gate for the real Davis DTI-to-draft workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from .review import ReviewRecord


REQUIRED_CLAIMS = {"C01", "C02"}
REQUIRED_SPLITS = ("pair_random", "cold_drug", "cold_target")
CLAIM_PATTERN = re.compile(r"\[@([A-Za-z][A-Za-z0-9_-]*)\]")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Release input is missing {label}.")
    return value


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Release input is missing numeric {label}.")
    return float(value)


def _validate_artifact_chain(model_report: Mapping[str, Any], ledger: Mapping[str, Any], draft_text: str,
                             trace: Mapping[str, Any]) -> list[dict[str, float | str]]:
    finalizable = set(ledger.get("finalizable_claim_ids", []))
    if not REQUIRED_CLAIMS <= finalizable:
        raise ValueError("Release requires finalizable verified context claims C01 and C02.")
    cited = set(CLAIM_PATTERN.findall(draft_text))
    if not REQUIRED_CLAIMS <= cited or not cited <= finalizable:
        raise ValueError("Draft claim references are incomplete or not finalizable.")
    if trace.get("dataset_id") != model_report.get("dataset_id") or trace.get("source_commit") != model_report.get("source_commit"):
        raise ValueError("Draft trace does not match the model-report provenance.")
    if set(trace.get("context_claim_ids", [])) != REQUIRED_CLAIMS:
        raise ValueError("Draft trace does not preserve the required context claims.")
    outcomes = _mapping(model_report.get("outcomes"), "model outcomes")
    trace_rows = trace.get("results")
    if not isinstance(trace_rows, list) or len(trace_rows) != len(REQUIRED_SPLITS):
        raise ValueError("Draft trace must contain all three recorded result conditions.")
    by_split = {row.get("split"): row for row in trace_rows if isinstance(row, Mapping)}
    released_rows = []
    for split in REQUIRED_SPLITS:
        outcome = _mapping(outcomes.get(split), f"{split} outcome")
        models = _mapping(outcome.get("models"), f"{split} models")
        forest = _mapping(models.get("random_forest"), f"{split} random forest")
        prevalence = _mapping(models.get("prevalence"), f"{split} prevalence")
        row = _mapping(by_split.get(split), f"{split} trace row")
        forest_pr = _number(forest.get("pr_auc"), f"{split} random-forest PR-AUC")
        prevalence_pr = _number(prevalence.get("pr_auc"), f"{split} prevalence PR-AUC")
        if _number(row.get("forest_pr_auc"), f"{split} traced random-forest PR-AUC") != forest_pr:
            raise ValueError(f"Draft trace disagrees with the model report for {split}.")
        if _number(row.get("prevalence_pr_auc"), f"{split} traced prevalence PR-AUC") != prevalence_pr:
            raise ValueError(f"Draft trace disagrees with the model report for {split}.")
        rendered = f"| {split.replace('_', ' ')} | {forest_pr:.3f} | {prevalence_pr:.3f} | {forest_pr - prevalence_pr:.3f} |"
        if rendered not in draft_text:
            raise ValueError(f"Draft does not contain the recorded {split} result row.")
        released_rows.append({"split": split, "forest_pr_auc": forest_pr,
                              "prevalence_pr_auc": prevalence_pr, "improvement": forest_pr - prevalence_pr})
    return released_rows


@dataclass(frozen=True)
class DavisReleaseReport:
    dataset_id: str
    source_commit: str
    context_claim_ids: tuple[str, ...]
    result_rows: tuple[dict[str, float | str], ...]
    tests_passed: bool
    compileall_passed: bool
    diff_check_passed: bool
    human_review_exportable: bool
    reviewer: str
    review_reason: str
    release_ready: bool
    limitations: tuple[str, ...]
    review_events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_v02_release(model_report: Mapping[str, Any], ledger: Mapping[str, Any], draft_text: str,
                         trace: Mapping[str, Any], *, reviewer: str, review_reason: str,
                         tests_passed: bool, compileall_passed: bool, diff_check_passed: bool) -> DavisReleaseReport:
    """Release only a trace-consistent draft that a named researcher approves and locks."""
    if not reviewer.strip() or not review_reason.strip():
        raise ValueError("A named reviewer and review reason are required for release.")
    rows = _validate_artifact_chain(model_report, ledger, draft_text, trace)
    dataset_id, source_commit = model_report.get("dataset_id"), model_report.get("source_commit")
    if not isinstance(dataset_id, str) or not isinstance(source_commit, str):
        raise ValueError("Model report requires dataset_id and source_commit.")
    review = ReviewRecord("v0_2_davis_constrained_draft")
    review.submit("approve", reviewer, review_reason, sorted(REQUIRED_CLAIMS))
    review.submit("lock", reviewer, "Approved draft locked for v0.2 release.", sorted(REQUIRED_CLAIMS))
    reproducible = bool(tests_passed and compileall_passed and diff_check_passed)
    limitations = (
        "Crossref verification establishes bibliographic identity, not scientific entailment.",
        "Pair-random results do not establish cold-start generalization.",
        "This release is a researcher-reviewed baseline draft, not a clinical or publication-ready claim.",
    )
    return DavisReleaseReport(
        dataset_id=dataset_id,
        source_commit=source_commit,
        context_claim_ids=tuple(sorted(REQUIRED_CLAIMS)),
        result_rows=tuple(rows),
        tests_passed=bool(tests_passed),
        compileall_passed=bool(compileall_passed),
        diff_check_passed=bool(diff_check_passed),
        human_review_exportable=review.exportable(),
        reviewer=reviewer.strip(),
        review_reason=review_reason.strip(),
        release_ready=bool(reproducible and review.exportable()),
        limitations=limitations,
        review_events=tuple(review.to_dict()["events"]),
    )
