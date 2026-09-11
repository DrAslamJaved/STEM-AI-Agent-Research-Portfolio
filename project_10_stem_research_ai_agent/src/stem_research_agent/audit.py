"""Deterministic draft audits; findings require human resolution."""
from dataclasses import asdict, dataclass
import re
from .evidence import EvidenceLedger

@dataclass(frozen=True)
class AuditFinding:
    code: str
    severity: str
    message: str

@dataclass
class AuditReport:
    findings: list[AuditFinding]
    def to_dict(self): return {"passed": not any(x.severity == "error" for x in self.findings),
                               "findings":[asdict(x) for x in self.findings]}

CLAIM_PATTERN = re.compile(r"\[([A-Za-z]\d+)\]")
METRIC_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?)")

def audit_draft(text: str, ledger: EvidenceLedger, analysis: dict[str, float], *, source_titles=()) -> AuditReport:
    findings = []
    finalizable = ledger.finalizable_claim_ids()
    for claim_id in sorted(set(CLAIM_PATTERN.findall(text))):
        if claim_id not in finalizable:
            findings.append(AuditFinding("unsupported_claim", "error", f"Claim {claim_id} is not finalizable."))
    for key, rendered in METRIC_PATTERN.findall(text):
        if key not in analysis:
            findings.append(AuditFinding("unknown_numeric_claim", "error", f"Reported value {key} is absent from analysis."))
        elif abs(float(rendered) - float(analysis[key])) > 0.005:
            findings.append(AuditFinding("numeric_mismatch", "error", f"Reported {key}={rendered} differs from saved analysis."))
    lower = text.lower()
    for title in source_titles:
        if len(title) >= 12 and title.lower() in lower:
            findings.append(AuditFinding("source_phrase_overlap", "warning", f"Draft contains an exact source-title phrase: {title!r}."))
    if not CLAIM_PATTERN.search(text):
        findings.append(AuditFinding("missing_claim_trace", "warning", "Draft has no claim identifiers."))
    return AuditReport(findings)
