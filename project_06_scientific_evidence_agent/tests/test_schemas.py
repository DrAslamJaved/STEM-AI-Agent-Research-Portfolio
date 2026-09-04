"""Tests for the traceable claim, citation, and decision objects."""

from __future__ import annotations

import pytest

from evidence_agent.data.schemas import AuditDecision, Citation, Claim, Verdict


def test_assertive_decision_requires_matching_citation() -> None:
    citation = Citation(doc_id=11, sentence_ids=(0, 2), stance=Verdict.SUPPORT)
    decision = AuditDecision(
        claim_id=1,
        verdict=Verdict.SUPPORT,
        confidence=0.9,
        citations=(citation,),
    )

    assert decision.citations == (citation,)


def test_no_evidence_decision_is_an_explicit_abstention() -> None:
    decision = AuditDecision(
        claim_id=1,
        verdict=Verdict.NO_EVIDENCE,
        confidence=0.4,
    )

    assert decision.citations == ()


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Claim(claim_id=1, text=" "), "non-empty"),
        (
            lambda: Citation(
                doc_id=1,
                sentence_ids=(0,),
                stance=Verdict.NO_EVIDENCE,
            ),
            "cannot be cited",
        ),
        (
            lambda: AuditDecision(
                claim_id=1,
                verdict=Verdict.SUPPORT,
                confidence=0.9,
            ),
            "require at least one citation",
        ),
    ],
)
def test_invalid_domain_objects_are_rejected(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
