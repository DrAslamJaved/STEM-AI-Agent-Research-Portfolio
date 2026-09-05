"""Deterministic paired-bootstrap confidence intervals over frozen dev claims.

Given two audited trace sets produced from the *same* frozen runtime trace
(for example the Phase 05 policy and a selected Phase 06 policy), this module
resamples claims with replacement and recomputes every metric from the
resample's pooled counts and labels -- never by averaging per-claim F1 values,
which would not equal the pooled F1 the point estimate reports. Each bootstrap
draw is given an occurrence-specific namespace so that a claim drawn twice in
one resample contributes two distinct entries, rather than collapsing into one
under set-based deduplication.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from evidence_agent.data.schemas import Verdict
from evidence_agent.verification.agent import VerificationTrace
from evidence_agent.verification.scifact import GoldClaimAnnotation, citations_to_sentence_keys


class BootstrapError(ValueError):
    """Raised when frozen traces or bootstrap settings are invalid."""


_METRIC_NAMES = (
    "citation_correctness_f1",
    "claim_macro_f1",
    "coverage",
    "evidence_sentence_f1",
    "faithfulness",
    "unsupported_assertion_rate",
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(2 * precision * recall, precision + recall)


@dataclass(frozen=True, slots=True)
class ClaimOutcome:
    """One claim's gold label and audited decision, stripped of claim_id.

    Sentence and citation keys deliberately omit the claim id: the bootstrap
    aggregator re-namespaces every key by its occurrence position in a
    resample, so the claim id itself must not already appear inside the key.
    """

    claim_id: int
    gold_verdict: Verdict
    predicted_verdict: Verdict
    predicted_sentence_keys: frozenset[tuple[int, int, Verdict]]
    gold_sentence_keys: frozenset[tuple[int, int, Verdict]]
    predicted_citation_keys: frozenset[tuple[int, tuple[int, ...], Verdict]]
    gold_citation_keys: frozenset[tuple[int, tuple[int, ...], Verdict]]

    @property
    def assertive(self) -> bool:
        return self.predicted_verdict is not Verdict.NO_EVIDENCE

    @property
    def grounded(self) -> bool:
        return bool(self.predicted_sentence_keys & self.gold_sentence_keys)

    @property
    def faithful(self) -> bool:
        return self.assertive and self.grounded

    @property
    def unsupported(self) -> bool:
        return self.assertive and (self.predicted_verdict is not self.gold_verdict or not self.grounded)


def _strip_claim_id(
    keys: frozenset[tuple[int, int, int, Verdict]]
) -> frozenset[tuple[int, int, Verdict]]:
    return frozenset((doc_id, sentence_id, stance) for _, doc_id, sentence_id, stance in keys)


def _citation_keys(
    citations: Sequence, claim_id: int
) -> frozenset[tuple[int, tuple[int, ...], Verdict]]:
    return frozenset((citation.doc_id, citation.sentence_ids, citation.stance) for citation in citations)


def build_claim_outcomes(
    traces: Sequence[VerificationTrace],
    gold_annotations: Mapping[int, GoldClaimAnnotation],
) -> tuple[ClaimOutcome, ...]:
    """Decompose frozen, audited traces into per-claim bootstrap primitives.

    Claims are returned sorted by claim id so that two outcome sequences built
    from the same claim population are directly zippable by position.
    """
    trace_by_claim = {trace.decision.claim_id: trace for trace in traces}
    missing = sorted(set(gold_annotations) - set(trace_by_claim))
    if missing:
        raise BootstrapError(f"Missing runtime decisions for claim IDs: {missing}.")

    outcomes: list[ClaimOutcome] = []
    for claim_id in sorted(gold_annotations):
        gold = gold_annotations[claim_id]
        decision = trace_by_claim[claim_id].decision
        outcomes.append(
            ClaimOutcome(
                claim_id=claim_id,
                gold_verdict=gold.verdict,
                predicted_verdict=decision.verdict,
                predicted_sentence_keys=_strip_claim_id(
                    citations_to_sentence_keys(decision.citations, claim_id)
                ),
                gold_sentence_keys=_strip_claim_id(
                    citations_to_sentence_keys(gold.citations, claim_id)
                ),
                predicted_citation_keys=_citation_keys(decision.citations, claim_id),
                gold_citation_keys=_citation_keys(gold.citations, claim_id),
            )
        )
    return tuple(outcomes)


def _macro_f1(actual: Sequence[Verdict], predicted: Sequence[Verdict]) -> float:
    f1_values: list[float] = []
    for label in Verdict:
        true_positive = sum(a is label and p is label for a, p in zip(actual, predicted))
        false_positive = sum(a is not label and p is label for a, p in zip(actual, predicted))
        false_negative = sum(a is label and p is not label for a, p in zip(actual, predicted))
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1_values.append(_f1(precision, recall))
    return sum(f1_values) / len(f1_values)


def _pooled_metrics(
    occurrence_claim_ids: Sequence[int],
    outcomes_by_claim: Mapping[int, ClaimOutcome],
) -> dict[str, float]:
    """Recompute every metric from one resample's pooled counts and labels."""
    actual_labels: list[Verdict] = []
    predicted_labels: list[Verdict] = []
    predicted_sentence_keys: set[tuple[int, int, int, Verdict]] = set()
    gold_sentence_keys: set[tuple[int, int, int, Verdict]] = set()
    predicted_citation_keys: set[tuple[int, int, tuple[int, ...], Verdict]] = set()
    gold_citation_keys: set[tuple[int, int, tuple[int, ...], Verdict]] = set()
    assertive_count = 0
    faithful_count = 0
    unsupported_count = 0

    for occurrence_index, claim_id in enumerate(occurrence_claim_ids):
        outcome = outcomes_by_claim[claim_id]
        actual_labels.append(outcome.gold_verdict)
        predicted_labels.append(outcome.predicted_verdict)
        predicted_sentence_keys.update(
            (occurrence_index, *key) for key in outcome.predicted_sentence_keys
        )
        gold_sentence_keys.update((occurrence_index, *key) for key in outcome.gold_sentence_keys)
        predicted_citation_keys.update(
            (occurrence_index, *key) for key in outcome.predicted_citation_keys
        )
        gold_citation_keys.update((occurrence_index, *key) for key in outcome.gold_citation_keys)
        if outcome.assertive:
            assertive_count += 1
            faithful_count += int(outcome.faithful)
            unsupported_count += int(outcome.unsupported)

    evidence_true_positive = len(predicted_sentence_keys & gold_sentence_keys)
    evidence_precision = _safe_ratio(evidence_true_positive, len(predicted_sentence_keys))
    evidence_recall = _safe_ratio(evidence_true_positive, len(gold_sentence_keys))
    citation_true_positive = len(predicted_citation_keys & gold_citation_keys)
    citation_precision = _safe_ratio(citation_true_positive, len(predicted_citation_keys))
    citation_recall = _safe_ratio(citation_true_positive, len(gold_citation_keys))

    return {
        "citation_correctness_f1": _f1(citation_precision, citation_recall),
        "claim_macro_f1": _macro_f1(actual_labels, predicted_labels),
        "coverage": _safe_ratio(assertive_count, len(occurrence_claim_ids)),
        "evidence_sentence_f1": _f1(evidence_precision, evidence_recall),
        "faithfulness": _safe_ratio(faithful_count, assertive_count),
        "unsupported_assertion_rate": _safe_ratio(unsupported_count, assertive_count),
    }


def paired_bootstrap_confidence_intervals(
    selected_outcomes: Sequence[ClaimOutcome],
    baseline_outcomes: Sequence[ClaimOutcome],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Bootstrap the selected-minus-baseline delta for six audit metrics.

    Both arms are resampled with the identical claim-id draw per replicate
    (a paired design): only the audited decisions differ between arms, so any
    difference in the resampled metric comes from the policy, not from
    resampling noise applied inconsistently.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise BootstrapError("resamples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise BootstrapError("seed must be a non-negative integer.")
    if not 0.0 < confidence_level < 1.0:
        raise BootstrapError("confidence_level must lie in (0, 1).")

    selected_by_claim = {outcome.claim_id: outcome for outcome in selected_outcomes}
    baseline_by_claim = {outcome.claim_id: outcome for outcome in baseline_outcomes}
    claim_ids = tuple(sorted(selected_by_claim))
    if claim_ids != tuple(sorted(baseline_by_claim)):
        raise BootstrapError("Selected and baseline outcomes must cover identical claim IDs.")
    for claim_id in claim_ids:
        if selected_by_claim[claim_id].gold_verdict != baseline_by_claim[claim_id].gold_verdict:
            raise BootstrapError(f"Gold verdict for claim {claim_id} differs between arms.")
    if not claim_ids:
        raise BootstrapError("At least one claim is required for bootstrap resampling.")

    claim_id_array = np.array(claim_ids)
    claim_count = len(claim_ids)
    rng = np.random.default_rng(seed)

    deltas: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    for _ in range(resamples):
        draw_indices = rng.integers(0, claim_count, size=claim_count)
        drawn_claim_ids = [int(claim_id_array[index]) for index in draw_indices]
        selected_metrics = _pooled_metrics(drawn_claim_ids, selected_by_claim)
        baseline_metrics = _pooled_metrics(drawn_claim_ids, baseline_by_claim)
        for name in _METRIC_NAMES:
            deltas[name].append(selected_metrics[name] - baseline_metrics[name])

    alpha = 1.0 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    metrics: dict[str, dict[str, float]] = {}
    for name in _METRIC_NAMES:
        values = np.asarray(deltas[name], dtype=float)
        metrics[name] = {
            "lower": float(np.percentile(values, lower_percentile)),
            "mean": float(values.mean()),
            "upper": float(np.percentile(values, upper_percentile)),
        }

    return {
        "claim_count": claim_count,
        "confidence_level": confidence_level,
        "metric": "selected_minus_phase_05_delta",
        "metrics": metrics,
        "resamples": resamples,
        "seed": seed,
    }
