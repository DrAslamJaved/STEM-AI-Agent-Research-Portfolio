from stem_research_agent.evidence import Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource, VerificationStatus, rank_sources, verify_source_metadata

def verified_source(source_id="S1"):
    raw = EvidenceSource(source_id, "Drug target baseline", "Drug target interaction baseline.", 2024, "10.1000/demo", "https://doi.org/10.1000/demo")
    return EvidenceSource(**{**raw.__dict__, "verification_status":verify_source_metadata(raw)})

def test_doi_metadata_can_be_locally_verified():
    assert verified_source().verification_status == VerificationStatus.VERIFIED

def test_malformed_source_is_rejected():
    source = EvidenceSource("S1", "", "", 2024, "bad", None)
    assert verify_source_metadata(source) == VerificationStatus.REJECTED

def test_only_verified_supporting_evidence_finalizes_claim():
    ledger = EvidenceLedger(); ledger.add_source(verified_source())
    ledger.add_claim(Claim("C1", "Validated claim", "background", ClaimStatus.HUMAN_APPROVED))
    ledger.link(EvidenceLink("C1", "S1", "Direct support", "supports"))
    assert ledger.finalizable_claim_ids() == {"C1"}

def test_inconclusive_source_cannot_finalize_claim():
    ledger = EvidenceLedger()
    ledger.add_source(EvidenceSource("S1", "Title", "Abstract", 2024, None, "https://example.org"))
    ledger.add_claim(Claim("C1", "Validated claim", "background", ClaimStatus.HUMAN_APPROVED))
    ledger.link(EvidenceLink("C1", "S1", "Direct support", "supports"))
    assert not ledger.finalizable_claim_ids()

def test_ranking_is_deterministic():
    first, second = verified_source("S2"), verified_source("S1")
    assert [x.source_id for x in rank_sources("drug target", [first, second])] == ["S1", "S2"]
