from stem_research_agent.crossref import CrossrefWork, normalize_doi, parse_crossref_work, retrieve_verified_source
from stem_research_agent.evidence import ClaimStatus, VerificationStatus
from stem_research_agent.literature_ledger import build_literature_ledger


def _work(doi="10.1038/nbt.1990", title="Comprehensive analysis of kinase inhibitor selectivity", year=2011):
    return CrossrefWork(doi, title, year, f"https://doi.org/{doi}")


def test_doi_normalization_and_crossref_response_parsing():
    payload = {"message": {"DOI": "10.1038/NBT.1990", "title": ["Comprehensive analysis of kinase inhibitor selectivity"],
                           "issued": {"date-parts": [[2011, 10, 30]]}, "URL": "https://doi.org/10.1038/nbt.1990"}}
    parsed = parse_crossref_work(payload)
    assert normalize_doi("https://doi.org/10.1038/NBT.1990") == parsed.doi == "10.1038/nbt.1990"
    assert parsed.year == 2011


def test_crossref_source_requires_matching_approved_metadata():
    source = retrieve_verified_source("S01", "10.1038/nbt.1990", expected_title="Comprehensive analysis of kinase inhibitor selectivity",
                                       expected_year=2011, fetcher=lambda _: _work())
    assert source.verification_status == VerificationStatus.VERIFIED
    try:
        retrieve_verified_source("S01", "10.1038/nbt.1990", expected_title="A different paper", fetcher=lambda _: _work())
    except ValueError as exc:
        assert "title" in str(exc)
    else:
        raise AssertionError("Expected title mismatch to be rejected.")


def test_verified_sources_and_human_approved_claims_become_finalizable():
    plan = {
        "sources": [{"source_id": "S01", "doi": "10.1038/nbt.1990"}],
        "claims": [{"claim_id": "C01", "text": "Approved context", "kind": "background", "status": "HUMAN_APPROVED"}],
        "links": [{"claim_id": "C01", "source_id": "S01", "rationale": "Direct metadata-backed source", "entailment": "supports"}],
    }
    ledger = build_literature_ledger(plan, retrieve=lambda spec: retrieve_verified_source(
        spec["source_id"], spec["doi"], fetcher=lambda _: _work()))
    assert ledger.claims["C01"].status == ClaimStatus.HUMAN_APPROVED
    assert ledger.finalizable_claim_ids() == {"C01"}


def test_ledger_rejects_link_to_unknown_source():
    plan = {"sources": [], "claims": [], "links": [{"claim_id": "C01", "source_id": "S01", "rationale": "x", "entailment": "supports"}]}
    try:
        build_literature_ledger(plan)
    except ValueError as exc:
        assert "known claims and sources" in str(exc)
    else:
        raise AssertionError("Expected an unknown evidence link to be rejected.")
