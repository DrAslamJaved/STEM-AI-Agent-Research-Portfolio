"""Project contracts that prevent gold SciFact annotations entering runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evidence_agent.data.schemas import Claim


GOLD_ONLY_CLAIM_FIELDS = frozenset({"evidence", "cited_doc_ids"})
RUNTIME_CLAIM_FIELDS = frozenset({"id", "claim"})


class LeakageContractError(ValueError):
    """Raised when evaluator-only annotations reach a runtime boundary."""


def runtime_claim_from_scifact(raw_claim: Mapping[str, Any]) -> Claim:
    """Create a safe runtime claim, deliberately discarding gold annotations."""
    try:
        claim_id = raw_claim["id"]
        claim_text = raw_claim["claim"]
    except KeyError as error:
        raise ValueError(f"SciFact claim is missing required field: {error.args[0]}") from error

    return Claim(claim_id=claim_id, text=claim_text)


def assert_runtime_payload_is_safe(payload: Mapping[str, Any]) -> None:
    """Reject a payload that includes any evaluator-only SciFact field."""
    forbidden = GOLD_ONLY_CLAIM_FIELDS.intersection(payload)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise LeakageContractError(
            f"Runtime payload contains evaluator-only field(s): {names}."
        )
