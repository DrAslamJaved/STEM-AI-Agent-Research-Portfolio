import pytest
from stem_research_agent.evidence import Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource, VerificationStatus
from stem_research_agent.manuscript import StyleProfile, generate_methods_section, generate_results_section

def ledger(status=ClaimStatus.HUMAN_APPROVED, verified=VerificationStatus.VERIFIED):
    value = EvidenceLedger()
    value.add_source(EvidenceSource("S1", "Title", "Abstract", 2024, "10.1000/demo", "https://doi.org/10.1000/demo", verified))
    value.add_claim(Claim("C1", "A controlled claim.", "method", status))
    value.link(EvidenceLink("C1", "S1", "Support.", "supports"))
    return value

def test_methods_draft_uses_finalizable_claim():
    draft = generate_methods_section(ledger(), ["C1"])
    assert "[C1] A controlled claim." in draft.text

def test_unverified_or_unapproved_claim_is_blocked():
    with pytest.raises(ValueError): generate_methods_section(ledger(verified=VerificationStatus.INCONCLUSIVE), ["C1"])

def test_results_require_analysis_values():
    with pytest.raises(ValueError): generate_results_section(ledger(), ["C1"], {})

def test_results_values_have_controlled_precision():
    draft = generate_results_section(ledger(), ["C1"], {"f1": .66666}, StyleProfile(result_precision=2))
    assert "f1=0.67" in draft.text

def test_upper_heading_style():
    assert "## METHODS" in generate_methods_section(ledger(), ["C1"], StyleProfile(section_heading_case="upper")).text
