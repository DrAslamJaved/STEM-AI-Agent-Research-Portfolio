"""Small validated domain objects shared across the future pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    """The only claim-level outcomes emitted by the agent."""

    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NO_EVIDENCE = "NO_EVIDENCE"


@dataclass(frozen=True, slots=True)
class Claim:
    """Safe runtime representation of one scientific claim."""

    claim_id: int
    text: str

    def __post_init__(self) -> None:
        if isinstance(self.claim_id, bool) or not isinstance(self.claim_id, int):
            raise ValueError("claim_id must be an integer.")
        if self.claim_id < 0:
            raise ValueError("claim_id must be non-negative.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("claim text must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class Citation:
    """A sentence-specific citation accepted as evidence for one stance."""

    doc_id: int
    sentence_ids: tuple[int, ...]
    stance: Verdict

    def __post_init__(self) -> None:
        if isinstance(self.doc_id, bool) or not isinstance(self.doc_id, int):
            raise ValueError("doc_id must be an integer.")
        if self.doc_id < 0:
            raise ValueError("doc_id must be non-negative.")
        if not self.sentence_ids:
            raise ValueError("a citation must contain at least one sentence id.")
        if tuple(sorted(set(self.sentence_ids))) != self.sentence_ids:
            raise ValueError("sentence_ids must be unique and sorted.")
        if any(
            isinstance(sentence_id, bool)
            or not isinstance(sentence_id, int)
            or sentence_id < 0
            for sentence_id in self.sentence_ids
        ):
            raise ValueError("sentence_ids must be non-negative integers.")
        if self.stance is Verdict.NO_EVIDENCE:
            raise ValueError("NO_EVIDENCE cannot be cited as affirmative evidence.")


@dataclass(frozen=True, slots=True)
class AuditDecision:
    """A traceable verdict with either matching evidence or an abstention."""

    claim_id: int
    verdict: Verdict
    confidence: float
    citations: tuple[Citation, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1].")
        if self.verdict is Verdict.NO_EVIDENCE and self.citations:
            raise ValueError("NO_EVIDENCE decisions cannot contain accepted citations.")
        if self.verdict is not Verdict.NO_EVIDENCE and not self.citations:
            raise ValueError("assertive decisions require at least one citation.")
        if any(citation.stance is not self.verdict for citation in self.citations):
            raise ValueError("citation stances must match the decision verdict.")
