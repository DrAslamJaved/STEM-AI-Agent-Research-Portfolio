"""Phase 4 evidence, citation-verification, and claim-ledger contracts."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import re

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.I)

class ClaimStatus(str, Enum):
    AGENT_PROPOSED = "AGENT_PROPOSED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    EXPERIMENTALLY_SUPPORTED = "EXPERIMENTALLY_SUPPORTED"
    REJECTED = "REJECTED"

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"

@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    title: str
    abstract: str
    year: int
    doi: str | None = None
    url: str | None = None
    verification_status: VerificationStatus = VerificationStatus.INCONCLUSIVE

@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    kind: str
    status: ClaimStatus = ClaimStatus.AGENT_PROPOSED

@dataclass(frozen=True)
class EvidenceLink:
    claim_id: str
    source_id: str
    rationale: str
    entailment: str

@dataclass
class EvidenceLedger:
    sources: dict[str, EvidenceSource] = field(default_factory=dict)
    claims: dict[str, Claim] = field(default_factory=dict)
    links: list[EvidenceLink] = field(default_factory=list)

    def add_source(self, source: EvidenceSource) -> None:
        if source.source_id in self.sources: raise ValueError("Duplicate source_id.")
        self.sources[source.source_id] = source

    def add_claim(self, claim: Claim) -> None:
        if claim.claim_id in self.claims: raise ValueError("Duplicate claim_id.")
        self.claims[claim.claim_id] = claim

    def link(self, link: EvidenceLink) -> None:
        if link.claim_id not in self.claims or link.source_id not in self.sources:
            raise ValueError("Links must reference known claims and sources.")
        if not link.rationale.strip() or link.entailment not in {"supports", "contradicts", "context"}:
            raise ValueError("Links require a rationale and valid entailment label.")
        self.links.append(link)

    def finalizable_claim_ids(self) -> set[str]:
        supported = {
            link.claim_id for link in self.links
            if link.entailment == "supports"
            and self.sources[link.source_id].verification_status == VerificationStatus.VERIFIED
        }
        return {claim_id for claim_id, claim in self.claims.items()
                if claim.status in {ClaimStatus.HUMAN_APPROVED, ClaimStatus.EXPERIMENTALLY_SUPPORTED}
                and claim_id in supported}

    def to_dict(self) -> dict:
        return {"sources":[asdict(x) for x in self.sources.values()],
                "claims":[asdict(x) for x in self.claims.values()],
                "links":[asdict(x) for x in self.links],
                "finalizable_claim_ids":sorted(self.finalizable_claim_ids())}

def verify_source_metadata(source: EvidenceSource) -> VerificationStatus:
    """Conservative local metadata validation; it is not web verification."""
    if not source.source_id.strip() or not source.title.strip() or not source.url:
        return VerificationStatus.REJECTED
    if source.year < 1665 or source.year > 2100:
        return VerificationStatus.REJECTED
    if source.doi and not DOI_PATTERN.fullmatch(source.doi.strip()):
        return VerificationStatus.REJECTED
    return VerificationStatus.VERIFIED if source.doi else VerificationStatus.INCONCLUSIVE

def rank_sources(query: str, sources: list[EvidenceSource]) -> list[EvidenceSource]:
    """Deterministic lexical ranking baseline; no network or vector database."""
    terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", query) if len(term) > 2}
    def score(source):
        text = f"{source.title} {source.abstract}".lower()
        return (-sum(term in text for term in terms), source.source_id)
    return sorted(sources, key=score)
