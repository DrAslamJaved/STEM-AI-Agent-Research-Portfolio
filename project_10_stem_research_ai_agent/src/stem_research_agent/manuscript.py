"""Controlled manuscript drafting from approved evidence and supplied results."""
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence
from .evidence import EvidenceLedger

@dataclass(frozen=True)
class StyleProfile:
    section_heading_case: str = "title"
    result_precision: int = 3
    terminology: str = "formal"

@dataclass(frozen=True)
class DraftArtifact:
    section: str
    text: str
    claim_ids: tuple[str, ...]
    analysis_keys: tuple[str, ...]
    style: StyleProfile
    def to_dict(self): return asdict(self)

def _heading(name: str, profile: StyleProfile) -> str:
    return name.upper() if profile.section_heading_case == "upper" else name.title()

def _finalizable(ledger: EvidenceLedger, claim_ids: Sequence[str]) -> None:
    permitted = ledger.finalizable_claim_ids()
    missing = set(claim_ids) - permitted
    if missing: raise ValueError(f"Claims are not approved and verified: {sorted(missing)}")

def generate_methods_section(ledger: EvidenceLedger, claim_ids: Sequence[str], profile=StyleProfile()) -> DraftArtifact:
    _finalizable(ledger, claim_ids)
    statements = [ledger.claims[claim_id].text for claim_id in claim_ids]
    text = f"## {_heading('Methods', profile)}\n\n" + "\n\n".join(f"[{claim_id}] {statement}" for claim_id, statement in zip(claim_ids, statements))
    return DraftArtifact("methods", text, tuple(claim_ids), (), profile)

def generate_results_section(ledger: EvidenceLedger, claim_ids: Sequence[str], analysis: Mapping[str, float], profile=StyleProfile()) -> DraftArtifact:
    _finalizable(ledger, claim_ids)
    if not analysis: raise ValueError("Results drafting requires saved analysis values.")
    values = "; ".join(f"{key}={value:.{profile.result_precision}f}" for key, value in sorted(analysis.items()))
    claims = "\n\n".join(f"[{claim_id}] {ledger.claims[claim_id].text}" for claim_id in claim_ids)
    text = f"## {_heading('Results', profile)}\n\nObserved analysis values: {values}.\n\n{claims}"
    return DraftArtifact("results", text, tuple(claim_ids), tuple(sorted(analysis)), profile)
