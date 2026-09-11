import json
from pathlib import Path
from stem_research_agent.audit import audit_draft
from stem_research_agent.evidence import Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource, VerificationStatus

ledger = EvidenceLedger(); ledger.add_source(EvidenceSource("S1","Validated baseline source","A",2024,"10.1000/a","https://doi.org/a",VerificationStatus.VERIFIED))
ledger.add_claim(Claim("C1","A transparent baseline was used.","method",ClaimStatus.HUMAN_APPROVED)); ledger.link(EvidenceLink("C1","S1","support","supports"))
report = audit_draft("## Results\n\naccuracy=0.75\n\n[C1] A transparent baseline was used.",ledger,{"accuracy":.75})
output=Path("results/phase_06_audit_demo.json"); output.write_text(json.dumps(report.to_dict(),indent=2)+"\n",encoding="utf-8"); print(output)
