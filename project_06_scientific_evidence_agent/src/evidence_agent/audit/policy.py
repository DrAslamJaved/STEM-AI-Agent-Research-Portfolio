"""Gold-free citation-audit policies applied to frozen verifier traces.

The verifier records stance probabilities and sentence scores for every
retrieved candidate.  This module can therefore re-apply a fixed confidence
policy without re-running retrieval or reading SciFact annotations.  Policy
selection belongs in a separate cross-validation phase.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from evidence_agent.data.schemas import AuditDecision, Citation, Verdict
from evidence_agent.verification.agent import CandidateTrace, VerificationTrace
from evidence_agent.verification.models import SentenceScore


class CitationAuditError(ValueError):
    """Raised when an audit policy or frozen trace is invalid."""


def _validate_probability(value: float, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise CitationAuditError(f"{name} must lie in [0, 1].")
    return float(value)


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CitationAuditError(f"{name} must be a positive integer.")
    return value


@dataclass(frozen=True, slots=True)
class CitationAuditPolicy:
    """A transparent abstention and sentence-citation acceptance rule."""

    assertion_threshold: float
    sentence_threshold: float
    max_sentences_per_citation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assertion_threshold",
            _validate_probability(self.assertion_threshold, "assertion_threshold"),
        )
        object.__setattr__(
            self,
            "sentence_threshold",
            _validate_probability(self.sentence_threshold, "sentence_threshold"),
        )
        _validate_positive_int(self.max_sentences_per_citation, "max_sentences_per_citation")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "assertion_threshold": self.assertion_threshold,
            "max_sentences_per_citation": self.max_sentences_per_citation,
            "sentence_threshold": self.sentence_threshold,
        }


PHASE_05_POLICY = CitationAuditPolicy(
    assertion_threshold=0.45,
    sentence_threshold=0.50,
    max_sentences_per_citation=2,
)


def _select_sentence_ids(
    scores: Sequence[SentenceScore],
    *,
    threshold: float,
    maximum: int,
) -> tuple[tuple[int, ...], float]:
    eligible = [score for score in scores if score.probability >= threshold]
    if not eligible:
        return (), 0.0
    selected = sorted(
        eligible,
        key=lambda score: (-score.probability, score.input.sentence_id),
    )[:maximum]
    return (
        tuple(sorted(score.input.sentence_id for score in selected)),
        selected[0].probability,
    )


def _audit_candidate(candidate: CandidateTrace, policy: CitationAuditPolicy) -> CandidateTrace:
    sentence_ids, sentence_probability = _select_sentence_ids(
        candidate.sentence_scores,
        threshold=policy.sentence_threshold,
        maximum=policy.max_sentences_per_citation,
    )
    stance_probability = candidate.stance.probability(candidate.selected_stance)
    combined_confidence = (
        math.sqrt(stance_probability * sentence_probability) if sentence_ids else 0.0
    )
    return replace(
        candidate,
        selected_sentence_ids=sentence_ids,
        selected_sentence_probability=sentence_probability,
        combined_confidence=combined_confidence,
    )


def _decision_from_candidates(
    claim_id: int,
    candidates: Sequence[CandidateTrace],
    *,
    assertion_threshold: float,
) -> AuditDecision:
    usable = [
        candidate
        for candidate in candidates
        if candidate.selected_sentence_ids
        and candidate.combined_confidence >= assertion_threshold
    ]
    if not usable:
        no_evidence_confidence = max(
            (candidate.stance.probability(Verdict.NO_EVIDENCE) for candidate in candidates),
            default=1.0,
        )
        return AuditDecision(
            claim_id=claim_id,
            verdict=Verdict.NO_EVIDENCE,
            confidence=no_evidence_confidence,
        )
    best = min(
        usable,
        key=lambda candidate: (
            -candidate.combined_confidence,
            -candidate.selected_sentence_probability,
            candidate.retrieval_rank,
            candidate.retrieval_hit.doc_id,
        ),
    )
    return AuditDecision(
        claim_id=claim_id,
        verdict=best.selected_stance,
        confidence=best.combined_confidence,
        citations=(
            Citation(
                doc_id=best.retrieval_hit.doc_id,
                sentence_ids=best.selected_sentence_ids,
                stance=best.selected_stance,
            ),
        ),
    )


def apply_citation_audit(
    trace: VerificationTrace,
    policy: CitationAuditPolicy,
) -> VerificationTrace:
    """Apply a fixed policy to one frozen, gold-free runtime trace."""
    audited_candidates = tuple(
        _audit_candidate(candidate, policy) for candidate in trace.candidates
    )
    return VerificationTrace(
        decision=_decision_from_candidates(
            trace.decision.claim_id,
            audited_candidates,
            assertion_threshold=policy.assertion_threshold,
        ),
        candidates=audited_candidates,
    )


def apply_citation_audit_to_traces(
    traces: Sequence[VerificationTrace],
    policy: CitationAuditPolicy,
) -> tuple[VerificationTrace, ...]:
    """Apply a fixed policy to a complete frozen runtime trace collection."""
    claim_ids = [trace.decision.claim_id for trace in traces]
    if len(claim_ids) != len(set(claim_ids)):
        raise CitationAuditError("Frozen traces must contain unique claim IDs.")
    return tuple(apply_citation_audit(trace, policy) for trace in traces)
