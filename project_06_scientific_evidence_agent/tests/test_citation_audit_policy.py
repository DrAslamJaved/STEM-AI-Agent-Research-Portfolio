"""Tests for gold-free citation-audit policy application."""

from __future__ import annotations

import pytest

from evidence_agent.audit.policy import (
    CitationAuditError,
    CitationAuditPolicy,
    apply_citation_audit,
    apply_citation_audit_to_traces,
)
from evidence_agent.data.schemas import AuditDecision, Citation, Verdict
from evidence_agent.retrieval.bm25 import RetrievalHit
from evidence_agent.verification.agent import CandidateTrace, VerificationTrace
from evidence_agent.verification.models import (
    SentenceInput,
    SentenceScore,
    StanceInput,
    StancePrediction,
)


def _trace() -> VerificationTrace:
    stance = StancePrediction(
        input=StanceInput(
            claim_id=1,
            doc_id=10,
            claim_text="A claim.",
            document_text="A public document.",
        ),
        verdict=Verdict.SUPPORT,
        probabilities=(
            (Verdict.SUPPORT, 0.81),
            (Verdict.CONTRADICT, 0.10),
            (Verdict.NO_EVIDENCE, 0.09),
        ),
    )
    scores = (
        SentenceScore(
            input=SentenceInput(1, 10, 0, "A claim.", "Weak candidate sentence."),
            probability=0.55,
        ),
        SentenceScore(
            input=SentenceInput(1, 10, 1, "A claim.", "Strong candidate sentence."),
            probability=0.90,
        ),
    )
    candidate = CandidateTrace(
        retrieval_rank=1,
        retrieval_hit=RetrievalHit(doc_id=10, score=1.0),
        stance=stance,
        sentence_scores=scores,
        selected_sentence_ids=(0, 1),
        selected_sentence_probability=0.90,
        selected_stance=Verdict.SUPPORT,
        combined_confidence=0.85,
    )
    return VerificationTrace(
        decision=AuditDecision(
            1,
            Verdict.SUPPORT,
            0.85,
            (Citation(10, (0, 1), Verdict.SUPPORT),),
        ),
        candidates=(candidate,),
    )


def test_policy_reselects_sentences_without_mutating_the_frozen_trace() -> None:
    frozen = _trace()

    audited = apply_citation_audit(
        frozen,
        CitationAuditPolicy(
            assertion_threshold=0.80,
            sentence_threshold=0.80,
            max_sentences_per_citation=1,
        ),
    )

    assert audited.decision.verdict is Verdict.SUPPORT
    assert audited.decision.citations[0].sentence_ids == (1,)
    assert audited.candidates[0].selected_sentence_ids == (1,)
    assert frozen.candidates[0].selected_sentence_ids == (0, 1)


def test_policy_abstains_when_no_sentence_passes_the_citation_threshold() -> None:
    audited = apply_citation_audit(
        _trace(),
        CitationAuditPolicy(
            assertion_threshold=0.0,
            sentence_threshold=0.95,
            max_sentences_per_citation=2,
        ),
    )

    assert audited.decision.verdict is Verdict.NO_EVIDENCE
    assert audited.decision.citations == ()


def test_policy_collection_rejects_duplicate_claim_decisions() -> None:
    with pytest.raises(CitationAuditError, match="unique claim IDs"):
        apply_citation_audit_to_traces(
            (_trace(), _trace()),
            CitationAuditPolicy(0.0, 0.0, 1),
        )
