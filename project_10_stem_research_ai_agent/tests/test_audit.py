from stem_research_agent.audit import audit_draft
from stem_research_agent.evidence import Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource, VerificationStatus

def ledger():
    value=EvidenceLedger();value.add_source(EvidenceSource("S1","Source title","A",2024,"10.1000/a","https://doi.org/a",VerificationStatus.VERIFIED))
    value.add_claim(Claim("C1","Claim","method",ClaimStatus.HUMAN_APPROVED));value.link(EvidenceLink("C1","S1","support","supports"));return value

def codes(report): return {x.code for x in report.findings}
def test_supported_claim_and_metric_pass(): assert audit_draft("[C1] accuracy=0.75",ledger(),{"accuracy":.75}).to_dict()["passed"]
def test_unknown_claim_is_error(): assert "unsupported_claim" in codes(audit_draft("[C9]",ledger(),{}))
def test_unknown_metric_is_error(): assert "unknown_numeric_claim" in codes(audit_draft("auc=0.8",ledger(),{}))
def test_mismatched_metric_is_error(): assert "numeric_mismatch" in codes(audit_draft("accuracy=0.6",ledger(),{"accuracy":.75}))
def test_exact_source_title_is_warning(): assert "source_phrase_overlap" in codes(audit_draft("[C1] Source title",ledger(),{},source_titles=["Source title"]))
