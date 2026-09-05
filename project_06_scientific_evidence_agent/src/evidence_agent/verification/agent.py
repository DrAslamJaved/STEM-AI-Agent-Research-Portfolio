"""Leakage-safe runtime orchestration for evidence selection and verification."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidence_agent.data.schemas import AuditDecision, Citation, Claim, Verdict
from evidence_agent.retrieval.bm25 import BM25Index, RetrievalHit
from evidence_agent.retrieval.scifact import CorpusDocument
from evidence_agent.verification.models import (
    SentenceInput,
    SentenceScore,
    StanceInput,
    StancePrediction,
    VerifierBundle,
)


DEFAULT_RETRIEVAL_K = 10
DEFAULT_ASSERTION_THRESHOLD = 0.45
DEFAULT_SENTENCE_THRESHOLD = 0.50
DEFAULT_MAX_SENTENCES_PER_CITATION = 2


class VerificationAgentError(ValueError):
    """Raised when a runtime evidence-verification request is invalid."""


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VerificationAgentError(f"{name} must be a positive integer.")


def _validate_probability(value: float, name: str) -> None:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise VerificationAgentError(f"{name} must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    """Runtime-only diagnostics for one retrieved document candidate."""

    retrieval_rank: int
    retrieval_hit: RetrievalHit
    stance: StancePrediction
    sentence_scores: tuple[SentenceScore, ...]
    selected_sentence_ids: tuple[int, ...]
    selected_sentence_probability: float
    selected_stance: Verdict
    combined_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "bm25_score": self.retrieval_hit.score,
            "combined_confidence": self.combined_confidence,
            "doc_id": self.retrieval_hit.doc_id,
            "retrieval_rank": self.retrieval_rank,
            "selected_sentence_ids": list(self.selected_sentence_ids),
            "selected_sentence_probability": self.selected_sentence_probability,
            "selected_stance": str(self.selected_stance),
            "sentence_scores": [score.as_dict() for score in self.sentence_scores],
            "stance": self.stance.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class VerificationTrace:
    """Frozen agent decision plus diagnostic data that contains no gold fields."""

    decision: AuditDecision
    candidates: tuple[CandidateTrace, ...]

    def decision_dict(self) -> dict[str, object]:
        """Return the compact, version-controlled decision record."""
        return {
            "citations": [
                {
                    "doc_id": citation.doc_id,
                    "sentence_ids": list(citation.sentence_ids),
                    "stance": str(citation.stance),
                }
                for citation in self.decision.citations
            ],
            "claim_id": self.decision.claim_id,
            "confidence": self.decision.confidence,
            "verdict": str(self.decision.verdict),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "decision": self.decision_dict(),
        }


def _select_sentence_ids(
    scores: Sequence[SentenceScore], *, threshold: float, maximum: int
) -> tuple[tuple[int, ...], float]:
    eligible = [score for score in scores if score.probability >= threshold]
    if not eligible:
        return (), 0.0
    selected = sorted(eligible, key=lambda score: (-score.probability, score.input.sentence_id))[ :maximum]
    return tuple(sorted(score.input.sentence_id for score in selected)), selected[0].probability


def _best_assertive_stance(prediction: StancePrediction) -> tuple[Verdict, float]:
    support = prediction.probability(Verdict.SUPPORT)
    contradict = prediction.probability(Verdict.CONTRADICT)
    if support >= contradict:
        return Verdict.SUPPORT, support
    return Verdict.CONTRADICT, contradict


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


def run_verification_agent(
    bundle: VerifierBundle,
    bm25_index: BM25Index,
    corpus: Mapping[int, CorpusDocument],
    claims: Sequence[Claim],
    *,
    retrieval_k: int = DEFAULT_RETRIEVAL_K,
    assertion_threshold: float = DEFAULT_ASSERTION_THRESHOLD,
    sentence_threshold: float = DEFAULT_SENTENCE_THRESHOLD,
    max_sentences_per_citation: int = DEFAULT_MAX_SENTENCES_PER_CITATION,
) -> tuple[VerificationTrace, ...]:
    """Freeze runtime decisions using only claims, corpus text, and local models.

    This function cannot receive SciFact ``evidence`` or ``cited_doc_ids``.
    Evaluation is a separate phase that consumes the returned immutable traces.
    """
    _validate_positive_int(retrieval_k, "retrieval_k")
    _validate_positive_int(max_sentences_per_citation, "max_sentences_per_citation")
    _validate_probability(assertion_threshold, "assertion_threshold")
    _validate_probability(sentence_threshold, "sentence_threshold")
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise VerificationAgentError("Runtime claim IDs must be unique.")
    if set(corpus) != set(bm25_index.document_lengths):
        raise VerificationAgentError("BM25 index and corpus must contain identical document IDs.")

    retrievals: dict[int, tuple[RetrievalHit, ...]] = {
        claim.claim_id: bm25_index.search(claim.text, retrieval_k) for claim in claims
    }
    stance_inputs: list[StanceInput] = []
    for claim in claims:
        for hit in retrievals[claim.claim_id]:
            stance_inputs.append(
                StanceInput(
                    claim_id=claim.claim_id,
                    doc_id=hit.doc_id,
                    claim_text=claim.text,
                    document_text=corpus[hit.doc_id].searchable_text,
                )
            )
    stance_predictions = bundle.predict_stances(stance_inputs)
    stance_by_pair = {
        (prediction.input.claim_id, prediction.input.doc_id): prediction
        for prediction in stance_predictions
    }

    sentence_inputs: list[SentenceInput] = []
    for claim in claims:
        for hit in retrievals[claim.claim_id]:
            for sentence_id, sentence_text in enumerate(corpus[hit.doc_id].abstract):
                sentence_inputs.append(
                    SentenceInput(
                        claim_id=claim.claim_id,
                        doc_id=hit.doc_id,
                        sentence_id=sentence_id,
                        claim_text=claim.text,
                        sentence_text=sentence_text,
                    )
                )
    sentence_scores = bundle.score_sentences(sentence_inputs)
    sentence_scores_by_pair: dict[tuple[int, int], list[SentenceScore]] = {}
    for score in sentence_scores:
        sentence_scores_by_pair.setdefault((score.input.claim_id, score.input.doc_id), []).append(score)

    traces: list[VerificationTrace] = []
    for claim in claims:
        candidates: list[CandidateTrace] = []
        for rank, hit in enumerate(retrievals[claim.claim_id], start=1):
            stance = stance_by_pair[(claim.claim_id, hit.doc_id)]
            candidate_scores = tuple(
                sorted(
                    sentence_scores_by_pair[(claim.claim_id, hit.doc_id)],
                    key=lambda score: score.input.sentence_id,
                )
            )
            selected_sentence_ids, selected_sentence_probability = _select_sentence_ids(
                candidate_scores,
                threshold=sentence_threshold,
                maximum=max_sentences_per_citation,
            )
            selected_stance, stance_probability = _best_assertive_stance(stance)
            combined_confidence = (
                math.sqrt(stance_probability * selected_sentence_probability)
                if selected_sentence_ids
                else 0.0
            )
            candidates.append(
                CandidateTrace(
                    retrieval_rank=rank,
                    retrieval_hit=hit,
                    stance=stance,
                    sentence_scores=candidate_scores,
                    selected_sentence_ids=selected_sentence_ids,
                    selected_sentence_probability=selected_sentence_probability,
                    selected_stance=selected_stance,
                    combined_confidence=combined_confidence,
                )
            )
        decision = _decision_from_candidates(
            claim.claim_id,
            candidates,
            assertion_threshold=assertion_threshold,
        )
        traces.append(VerificationTrace(decision=decision, candidates=tuple(candidates)))
    return tuple(traces)
