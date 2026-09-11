import json
from pathlib import Path
from stem_research_agent.evidence import Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource, VerificationStatus
from stem_research_agent.manuscript import StyleProfile, generate_results_section

ledger = EvidenceLedger()
ledger.add_source(EvidenceSource("S01", "Validated baseline", "Abstract", 2024, "10.1000/demo", "https://doi.org/10.1000/demo", VerificationStatus.VERIFIED))
ledger.add_claim(Claim("C01", "The analysis used a transparent prevalence baseline.", "method", ClaimStatus.HUMAN_APPROVED))
ledger.link(EvidenceLink("C01", "S01", "Documents the baseline.", "supports"))
artifact = generate_results_section(ledger, ["C01"], {"accuracy": .75, "f1": .6}, StyleProfile(result_precision=2))
output = Path("results/phase_05_drafting_demo.json"); output.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n", encoding="utf-8")
print(output)
