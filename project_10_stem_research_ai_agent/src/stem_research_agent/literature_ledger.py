"""Build a researcher-approved claim ledger from Crossref-verified sources."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable, Mapping

from .crossref import retrieve_verified_source
from .evidence import Claim, ClaimStatus, EvidenceLedger, EvidenceLink, EvidenceSource


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Literature plan requires non-empty {field!r}.")
    return value.strip()


def load_literature_plan(path: str | Path) -> Mapping[str, Any]:
    """Load a small declarative plan; it contains claims, not generated prose."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load literature plan: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Literature plan must be a JSON object.")
    return value


def build_literature_ledger(plan: Mapping[str, Any], *,
                            retrieve: Callable[[Mapping[str, Any]], EvidenceSource] | None = None) -> EvidenceLedger:
    """Verify planned DOI sources and link them to explicitly approved claims."""
    sources, claims, links = plan.get("sources"), plan.get("claims"), plan.get("links")
    if not all(isinstance(value, list) for value in (sources, claims, links)):
        raise ValueError("Literature plan requires sources, claims, and links lists.")
    ledger = EvidenceLedger()
    for spec in sources:
        if not isinstance(spec, Mapping):
            raise ValueError("Each literature source must be an object.")
        if retrieve:
            source = retrieve(spec)
        else:
            source = retrieve_verified_source(
                _required_text(spec, "source_id"), _required_text(spec, "doi"),
                expected_title=spec.get("expected_title"), expected_year=spec.get("expected_year"),
            )
        ledger.add_source(source)
    for spec in claims:
        if not isinstance(spec, Mapping):
            raise ValueError("Each literature claim must be an object.")
        try:
            status = ClaimStatus(_required_text(spec, "status"))
        except ValueError as exc:
            raise ValueError("Claim status must be an approved evidence-ledger status.") from exc
        ledger.add_claim(Claim(_required_text(spec, "claim_id"), _required_text(spec, "text"),
                                _required_text(spec, "kind"), status))
    for spec in links:
        if not isinstance(spec, Mapping):
            raise ValueError("Each literature link must be an object.")
        ledger.link(EvidenceLink(_required_text(spec, "claim_id"), _required_text(spec, "source_id"),
                                 _required_text(spec, "rationale"), _required_text(spec, "entailment")))
    return ledger
