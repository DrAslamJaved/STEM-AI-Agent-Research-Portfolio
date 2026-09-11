import json
from pathlib import Path
from stem_research_agent.evidence import (
    Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource,
    VerificationStatus, rank_sources, verify_source_metadata,
)

source = EvidenceSource("S01", "Drug target prediction baseline", "A transparent baseline for drug target interaction prediction.", 2024,
    "10.1000/example.dti", "https://doi.org/10.1000/example.dti")
source = EvidenceSource(**{**source.__dict__, "verification_status": verify_source_metadata(source)})
claim = Claim("C01", "A prevalence baseline is a transparent reference for binary DTI prediction.", "method", ClaimStatus.HUMAN_APPROVED)
ledger = EvidenceLedger(); ledger.add_source(source); ledger.add_claim(claim)
ledger.link(EvidenceLink("C01", "S01", "The source describes baseline DTI prediction.", "supports"))
result = {"ranking":[x.source_id for x in rank_sources("transparent drug target baseline", [source])], "ledger":ledger.to_dict()}
output = Path("results/phase_04_evidence_demo.json"); output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
print(output)
