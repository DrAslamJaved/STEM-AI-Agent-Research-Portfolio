"""Tests for stance, evidence, citation, and unsupported-claim metrics."""

from __future__ import annotations

from evidence_agent.data.schemas import AuditDecision, Citation, Verdict
from evidence_agent.evaluation.verification import (
    evaluate_stance_benchmark,
    evaluate_verification_traces,
)
from evidence_agent.verification.agent import VerificationTrace
from evidence_agent.verification.models import StanceInput, StancePrediction
from evidence_agent.verification.scifact import GoldClaimAnnotation


def _prediction(claim_id: int, verdict: Verdict) -> StancePrediction:
    return StancePrediction(
        input=StanceInput(
            claim_id=claim_id,
            doc_id=claim_id + 10,
            claim_text="A claim.",
            document_text="A document.",
        ),
        verdict=verdict,
        probabilities=tuple(
            (label, 1.0 if label is verdict else 0.0) for label in Verdict
        ),
    )


def test_stance_benchmark_reports_perfect_three_way_macro_f1() -> None:
    result = evaluate_stance_benchmark(
        tuple(_prediction(index, label) for index, label in enumerate(Verdict)),
        tuple(Verdict),
    )

    assert result.summary_dict()["macro_f1"] == 1.0
    assert result.summary_dict()["pair_count"] == 3


def test_agent_evaluation_measures_evidence_citations_and_unsupported_rate() -> None:
    support_citation = Citation(doc_id=10, sentence_ids=(0,), stance=Verdict.SUPPORT)
    contradict_citation = Citation(doc_id=11, sentence_ids=(1,), stance=Verdict.CONTRADICT)
    traces = (
        VerificationTrace(
            decision=AuditDecision(1, Verdict.SUPPORT, 0.9, (support_citation,)),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(2, Verdict.CONTRADICT, 0.9, (contradict_citation,)),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(3, Verdict.NO_EVIDENCE, 0.9),
            candidates=(),
        ),
    )
    gold = {
        1: GoldClaimAnnotation(1, Verdict.SUPPORT, (support_citation,)),
        2: GoldClaimAnnotation(2, Verdict.CONTRADICT, (contradict_citation,)),
        3: GoldClaimAnnotation(3, Verdict.NO_EVIDENCE, ()),
    }

    result = evaluate_verification_traces(traces, gold).summary_dict()

    assert result["claim_classification"]["macro_f1"] == 1.0
    assert result["evidence_sentence"]["f1"] == 1.0
    assert result["citation_correctness"]["f1"] == 1.0
    assert result["faithfulness"] == 1.0
    assert result["unsupported_assertion_rate"] == 0.0
