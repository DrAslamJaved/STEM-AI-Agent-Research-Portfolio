import pytest

from stem_research_agent.davis_manuscript import build_constrained_davis_draft
from stem_research_agent.v02_release import evaluate_v02_release


def _ledger():
    return {
        "finalizable_claim_ids": ["C01", "C02"],
        "sources": [{"source_id": "S01"}, {"source_id": "S02"}],
        "links": [{"claim_id": "C01", "source_id": "S01", "entailment": "supports"},
                  {"claim_id": "C02", "source_id": "S02", "entailment": "supports"}],
    }


def _draft_ledger():
    ledger = _ledger()
    ledger["sources"] = [
        {"source_id": "S01", "title": "Davis", "year": 2011, "doi": "10.1038/nbt.1990", "verification_status": "VERIFIED"},
        {"source_id": "S02", "title": "DeepDTA", "year": 2018, "doi": "10.1093/bioinformatics/bty593", "verification_status": "VERIFIED"},
    ]
    return ledger


def _model_report():
    def outcome(forest, baseline):
        return {"models": {"random_forest": {"pr_auc": forest}, "prevalence": {"pr_auc": baseline}}}
    return {"dataset_id": "davis_demo", "source_commit": "abc123", "outcomes": {
        "pair_random": outcome(.61, .08), "cold_drug": outcome(.40, .09), "cold_target": outcome(.58, .10)}}


def _inputs():
    report, ledger = _model_report(), _draft_ledger()
    draft = build_constrained_davis_draft(report, ledger)
    return report, ledger, draft.markdown, draft.trace


def test_release_is_ready_after_trace_validation_and_named_review():
    result = evaluate_v02_release(*_inputs(), reviewer="Researcher", review_reason="Reviewed evidence and result trace.",
                                  tests_passed=True, compileall_passed=True, diff_check_passed=True)
    assert result.release_ready and result.human_review_exportable
    assert [event["action"] for event in result.review_events] == ["approve", "lock"]


def test_release_is_blocked_when_reproducibility_evidence_is_missing():
    result = evaluate_v02_release(*_inputs(), reviewer="Researcher", review_reason="Reviewed.",
                                  tests_passed=True, compileall_passed=False, diff_check_passed=True)
    assert not result.release_ready


def test_release_rejects_draft_with_unfinalizable_claim():
    report, ledger, draft, trace = _inputs()
    with pytest.raises(ValueError, match="not finalizable"):
        evaluate_v02_release(report, ledger, draft + "\nUnsupported [@C99].", trace, reviewer="Researcher",
                             review_reason="Reviewed.", tests_passed=True, compileall_passed=True, diff_check_passed=True)


def test_release_rejects_trace_that_disagrees_with_model_report():
    report, ledger, draft, trace = _inputs()
    trace["results"][0]["forest_pr_auc"] = .5
    with pytest.raises(ValueError, match="disagrees"):
        evaluate_v02_release(report, ledger, draft, trace, reviewer="Researcher", review_reason="Reviewed.",
                             tests_passed=True, compileall_passed=True, diff_check_passed=True)
