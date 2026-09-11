from types import SimpleNamespace

import pytest

from stem_research_agent.release import evaluate_draft, evaluate_release
from stem_research_agent.review import ReviewRecord


def _approved_review():
    review = ReviewRecord("D1")
    review.submit("approve", "Reviewer", "Evidence checked", ["C01"])
    review.submit("lock", "Reviewer", "Final lock", ["C01"])
    return review


def _report(**overrides):
    defaults = dict(
        baseline_text="Ungrounded comparison [C99].",
        controlled_text="Evidence-grounded comparison [C01].",
        ledger=SimpleNamespace(finalizable_claim_ids=["C01"]),
        audit_report=SimpleNamespace(passed=True, findings=[]),
        review_record=_approved_review(),
        tests_passed=True,
        compileall_passed=True,
        diff_check_passed=True,
        human_correction_count=2,
    )
    defaults.update(overrides)
    return evaluate_release(**defaults)


def test_draft_evaluation_counts_only_finalizable_claims():
    result = evaluate_draft("Supported [C01]; unsupported [C02].", SimpleNamespace(finalizable_claim_ids=["C01"]))
    assert result.supported_claim_ids == ("C01",)
    assert result.unsupported_claim_ids == ("C02",)
    assert result.supported_claim_rate == 0.5


def test_controlled_draft_outperforms_ungrounded_baseline():
    result = _report()
    assert result.baseline.supported_claim_rate == 0.0
    assert result.controlled.supported_claim_rate == 1.0
    assert result.supported_claim_rate_delta == 1.0
    assert result.release_ready


def test_release_is_blocked_when_reproducibility_gate_fails():
    result = _report(compileall_passed=False)
    assert not result.reproducibility_passed
    assert not result.release_ready


def test_release_is_blocked_without_exportable_human_review():
    review = ReviewRecord("D1")
    review.submit("approve", "Reviewer", "Checked", ["C01"])
    result = _report(review_record=review)
    assert not result.human_review_exportable
    assert not result.release_ready


def test_negative_human_correction_count_is_rejected():
    with pytest.raises(ValueError):
        _report(human_correction_count=-1)
