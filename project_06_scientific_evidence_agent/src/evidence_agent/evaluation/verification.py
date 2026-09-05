"""Leakage-safe metrics for stance, rationale selection, and claim decisions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_agent.data.schemas import Citation, Verdict
from evidence_agent.verification.agent import VerificationTrace
from evidence_agent.verification.models import StancePrediction
from evidence_agent.verification.scifact import (
    GoldClaimAnnotation,
    citations_to_sentence_keys,
)


class VerificationEvaluationError(ValueError):
    """Raised when frozen verifier outputs and gold annotations disagree."""


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(2 * precision * recall, precision + recall)


def _classification_summary(
    actual: Sequence[Verdict], predicted: Sequence[Verdict]
) -> dict[str, object]:
    if len(actual) != len(predicted):
        raise VerificationEvaluationError("Actual and predicted label counts differ.")
    if not actual:
        raise VerificationEvaluationError("At least one labelled prediction is required.")
    per_label: dict[str, dict[str, float | int]] = {}
    confusion = {
        str(actual_label): {str(predicted_label): 0 for predicted_label in Verdict}
        for actual_label in Verdict
    }
    for actual_label, predicted_label in zip(actual, predicted, strict=True):
        confusion[str(actual_label)][str(predicted_label)] += 1
    f1_values: list[float] = []
    for label in Verdict:
        true_positive = sum(
            actual_label is label and predicted_label is label
            for actual_label, predicted_label in zip(actual, predicted, strict=True)
        )
        false_positive = sum(
            actual_label is not label and predicted_label is label
            for actual_label, predicted_label in zip(actual, predicted, strict=True)
        )
        false_negative = sum(
            actual_label is label and predicted_label is not label
            for actual_label, predicted_label in zip(actual, predicted, strict=True)
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        label_f1 = _f1(precision, recall)
        f1_values.append(label_f1)
        per_label[str(label)] = {
            "f1": label_f1,
            "precision": precision,
            "recall": recall,
            "support": sum(actual_label is label for actual_label in actual),
        }
    return {
        "accuracy": _safe_ratio(
            sum(
                actual_label is predicted_label
                for actual_label, predicted_label in zip(actual, predicted, strict=True)
            ),
            len(actual),
        ),
        "confusion_matrix": confusion,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label,
    }


@dataclass(frozen=True, slots=True)
class StanceBenchmarkResult:
    """Three-way macro-F1 on cited document pairs after predictions freeze."""

    pair_count: int
    summary: Mapping[str, object]

    def summary_dict(self) -> dict[str, object]:
        return {"pair_count": self.pair_count, **self.summary}


@dataclass(frozen=True, slots=True)
class EvidenceVerificationResult:
    """Claim-level, citation, and evidence-sentence metrics for frozen traces."""

    claim_count: int
    claim_summary: Mapping[str, object]
    evidence_sentence_precision: float
    evidence_sentence_recall: float
    evidence_sentence_f1: float
    strict_citation_precision: float
    strict_citation_recall: float
    strict_citation_f1: float
    coverage: float
    assertive_decision_count: int
    faithful_assertive_decision_count: int
    unsupported_assertive_decision_count: int

    def summary_dict(self) -> dict[str, object]:
        assertive_count = self.assertive_decision_count
        return {
            "assertive_decision_count": assertive_count,
            "citation_correctness": {
                "f1": self.strict_citation_f1,
                "precision": self.strict_citation_precision,
                "recall": self.strict_citation_recall,
            },
            "claim_classification": self.claim_summary,
            "claim_count": self.claim_count,
            "coverage": self.coverage,
            "evidence_sentence": {
                "f1": self.evidence_sentence_f1,
                "precision": self.evidence_sentence_precision,
                "recall": self.evidence_sentence_recall,
            },
            "faithfulness": _safe_ratio(
                self.faithful_assertive_decision_count, assertive_count
            ),
            "faithful_assertive_decision_count": self.faithful_assertive_decision_count,
            "unsupported_assertion_rate": _safe_ratio(
                self.unsupported_assertive_decision_count, assertive_count
            ),
            "unsupported_assertive_decision_count": self.unsupported_assertive_decision_count,
        }


def evaluate_stance_benchmark(
    predictions: Sequence[StancePrediction], gold_labels: Sequence[Verdict]
) -> StanceBenchmarkResult:
    """Score cited-document stance inputs after the model emitted all outputs."""
    if len(predictions) != len(gold_labels):
        raise VerificationEvaluationError("Stance prediction and gold-label counts differ.")
    if not predictions:
        raise VerificationEvaluationError("No stance predictions are available for evaluation.")
    return StanceBenchmarkResult(
        pair_count=len(predictions),
        summary=_classification_summary(gold_labels, [prediction.verdict for prediction in predictions]),
    )


def _citation_key(claim_id: int, citation: Citation) -> tuple[int, int, tuple[int, ...], Verdict]:
    return (claim_id, citation.doc_id, citation.sentence_ids, citation.stance)


def evaluate_verification_traces(
    traces: Sequence[VerificationTrace],
    gold_annotations: Mapping[int, GoldClaimAnnotation],
) -> EvidenceVerificationResult:
    """Evaluate frozen agent traces against evaluator-only gold annotations."""
    trace_by_claim: dict[int, VerificationTrace] = {}
    for trace in traces:
        claim_id = trace.decision.claim_id
        if claim_id in trace_by_claim:
            raise VerificationEvaluationError(f"Duplicate runtime decision for claim {claim_id}.")
        trace_by_claim[claim_id] = trace
    missing = sorted(set(gold_annotations) - set(trace_by_claim))
    if missing:
        raise VerificationEvaluationError(f"Missing runtime decisions for claim IDs: {missing}.")
    unexpected = sorted(set(trace_by_claim) - set(gold_annotations))
    if unexpected:
        raise VerificationEvaluationError(f"Runtime decisions have unknown claim IDs: {unexpected}.")

    claim_actual: list[Verdict] = []
    claim_predicted: list[Verdict] = []
    gold_citation_keys: set[tuple[int, int, tuple[int, ...], Verdict]] = set()
    predicted_citation_keys: set[tuple[int, int, tuple[int, ...], Verdict]] = set()
    gold_sentence_keys: set[tuple[int, int, int, Verdict]] = set()
    predicted_sentence_keys: set[tuple[int, int, int, Verdict]] = set()
    assertive_count = 0
    faithful_assertive_count = 0
    unsupported_assertive_count = 0

    for claim_id in sorted(gold_annotations):
        gold = gold_annotations[claim_id]
        trace = trace_by_claim[claim_id]
        decision = trace.decision
        claim_actual.append(gold.verdict)
        claim_predicted.append(decision.verdict)

        current_gold_sentence_keys = citations_to_sentence_keys(gold.citations, claim_id)
        current_predicted_sentence_keys = citations_to_sentence_keys(decision.citations, claim_id)
        gold_sentence_keys.update(current_gold_sentence_keys)
        predicted_sentence_keys.update(current_predicted_sentence_keys)
        gold_citation_keys.update(_citation_key(claim_id, citation) for citation in gold.citations)
        predicted_citation_keys.update(
            _citation_key(claim_id, citation) for citation in decision.citations
        )

        if decision.verdict is not Verdict.NO_EVIDENCE:
            assertive_count += 1
            grounded = bool(current_predicted_sentence_keys.intersection(current_gold_sentence_keys))
            faithful_assertive_count += int(grounded)
            unsupported_assertive_count += int(
                decision.verdict is not gold.verdict or not grounded
            )

    evidence_true_positive = len(predicted_sentence_keys.intersection(gold_sentence_keys))
    evidence_precision = _safe_ratio(evidence_true_positive, len(predicted_sentence_keys))
    evidence_recall = _safe_ratio(evidence_true_positive, len(gold_sentence_keys))
    strict_true_positive = len(predicted_citation_keys.intersection(gold_citation_keys))
    strict_precision = _safe_ratio(strict_true_positive, len(predicted_citation_keys))
    strict_recall = _safe_ratio(strict_true_positive, len(gold_citation_keys))

    return EvidenceVerificationResult(
        claim_count=len(gold_annotations),
        claim_summary=_classification_summary(claim_actual, claim_predicted),
        evidence_sentence_precision=evidence_precision,
        evidence_sentence_recall=evidence_recall,
        evidence_sentence_f1=_f1(evidence_precision, evidence_recall),
        strict_citation_precision=strict_precision,
        strict_citation_recall=strict_recall,
        strict_citation_f1=_f1(strict_precision, strict_recall),
        coverage=_safe_ratio(assertive_count, len(gold_annotations)),
        assertive_decision_count=assertive_count,
        faithful_assertive_decision_count=faithful_assertive_count,
        unsupported_assertive_decision_count=unsupported_assertive_count,
    )


def write_verification_report(payload: Mapping[str, object], path: Path) -> None:
    """Write a deterministic machine-readable verification report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
