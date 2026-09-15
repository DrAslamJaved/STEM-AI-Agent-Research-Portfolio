from stem_research_agent.davis_manuscript import build_constrained_davis_draft


def _ledger(finalizable=("C01", "C02")):
    return {
        "finalizable_claim_ids": list(finalizable),
        "sources": [
            {"source_id": "S01", "title": "Davis source", "year": 2011, "doi": "10.1038/nbt.1990", "verification_status": "VERIFIED"},
            {"source_id": "S02", "title": "DeepDTA source", "year": 2018, "doi": "10.1093/bioinformatics/bty593", "verification_status": "VERIFIED"},
        ],
        "links": [
            {"claim_id": "C01", "source_id": "S01", "entailment": "supports"},
            {"claim_id": "C02", "source_id": "S02", "entailment": "supports"},
        ],
    }


def _report():
    def outcome(forest, baseline):
        return {"models": {"random_forest": {"pr_auc": forest}, "prevalence": {"pr_auc": baseline}}}
    return {"dataset_id": "davis_demo", "source_commit": "abc123", "outcomes": {
        "pair_random": outcome(.61, .08), "cold_drug": outcome(.40, .09), "cold_target": outcome(.58, .10)}}


def test_draft_contains_approved_context_and_recorded_metrics():
    draft = build_constrained_davis_draft(_report(), _ledger())
    assert "[@C01]" in draft.markdown and "[@C02]" in draft.markdown
    assert "| pair random | 0.610 | 0.080 | 0.530 |" in draft.markdown
    assert draft.trace["context_claim_ids"] == ["C01", "C02"]


def test_draft_rejects_unapproved_context_claims():
    try:
        build_constrained_davis_draft(_report(), _ledger(("C01",)))
    except ValueError as exc:
        assert "C02" in str(exc)
    else:
        raise AssertionError("Expected missing finalizable claim to block drafting.")


def test_draft_requires_all_leakage_aware_split_conditions():
    report = _report(); del report["outcomes"]["cold_target"]
    try:
        build_constrained_davis_draft(report, _ledger())
    except ValueError as exc:
        assert "cold_target" in str(exc)
    else:
        raise AssertionError("Expected missing split to block drafting.")


def test_draft_rejects_invalid_recorded_metric():
    report = _report(); report["outcomes"]["cold_drug"]["models"]["random_forest"]["pr_auc"] = 1.1
    try:
        build_constrained_davis_draft(report, _ledger())
    except ValueError as exc:
        assert "invalid PR-AUC" in str(exc)
    else:
        raise AssertionError("Expected invalid model metric to block drafting.")
