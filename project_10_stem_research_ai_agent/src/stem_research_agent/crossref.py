"""Small, auditable Crossref DOI metadata client for v0.2 literature checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen

from .evidence import DOI_PATTERN, EvidenceSource, VerificationStatus


_DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.I)
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CrossrefWork:
    """The small bibliographic subset needed to verify a DOI identity."""

    doi: str
    title: str
    year: int
    url: str
    publisher: str | None = None


def normalize_doi(value: str) -> str:
    """Return a canonical DOI string and reject malformed identifiers."""
    doi = _DOI_URL_PREFIX.sub("", value.strip()).lower()
    if not DOI_PATTERN.fullmatch(doi):
        raise ValueError("A valid DOI is required for Crossref verification.")
    return doi


def _publication_year(message: Mapping[str, Any]) -> int:
    for key in ("published-print", "published-online", "issued", "created"):
        date_parts = message.get(key, {}).get("date-parts", [])
        if date_parts and date_parts[0] and isinstance(date_parts[0][0], int):
            return date_parts[0][0]
    raise ValueError("Crossref work does not provide a publication year.")


def parse_crossref_work(payload: Mapping[str, Any]) -> CrossrefWork:
    """Parse one Crossref `/works/{doi}` response without trusting extra fields."""
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Crossref response is missing its message object.")
    titles = message.get("title")
    if not isinstance(titles, list) or not titles or not isinstance(titles[0], str) or not titles[0].strip():
        raise ValueError("Crossref response is missing a title.")
    raw_doi = message.get("DOI")
    if not isinstance(raw_doi, str):
        raise ValueError("Crossref response is missing a DOI.")
    doi = normalize_doi(raw_doi)
    url = message.get("URL") or f"https://doi.org/{doi}"
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Crossref response is missing a canonical URL.")
    publisher = message.get("publisher")
    return CrossrefWork(doi, titles[0].strip(), _publication_year(message), url.strip(),
                         publisher.strip() if isinstance(publisher, str) and publisher.strip() else None)


def fetch_crossref_work(doi: str, *, mailto: str | None = None,
                        opener: Callable[..., Any] = urlopen, timeout: int = 20) -> CrossrefWork:
    """Retrieve Crossref metadata for one DOI; no article full text is fetched."""
    canonical_doi = normalize_doi(doi)
    user_agent = "stem-research-ai-agent/0.2"
    if mailto:
        user_agent += f" (mailto:{mailto})"
    request = Request(
        f"https://api.crossref.org/works/{canonical_doi}",
        headers={"Accept": "application/json", "User-Agent": user_agent},
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Crossref lookup failed for DOI {canonical_doi}: {exc}") from exc
    return parse_crossref_work(payload)


def _title_tokens(title: str) -> set[str]:
    return {word for word in _WORD.findall(title.lower()) if len(word) >= 3}


def title_matches(expected: str, observed: str) -> bool:
    """Require substantial token overlap while allowing harmless punctuation changes."""
    expected_terms = _title_tokens(expected)
    observed_terms = _title_tokens(observed)
    return bool(expected_terms) and len(expected_terms & observed_terms) / len(expected_terms) >= 0.8


def retrieve_verified_source(source_id: str, doi: str, *, expected_title: str | None = None,
                             expected_year: int | None = None, mailto: str | None = None,
                             fetcher: Callable[[str], CrossrefWork] | None = None) -> EvidenceSource:
    """Create an evidence source only after DOI identity checks against Crossref."""
    canonical_doi = normalize_doi(doi)
    work = fetcher(canonical_doi) if fetcher else fetch_crossref_work(canonical_doi, mailto=mailto)
    if work.doi != canonical_doi:
        raise ValueError("Crossref returned metadata for a different DOI.")
    if expected_title and not title_matches(expected_title, work.title):
        raise ValueError("Crossref title does not sufficiently match the approved title.")
    if expected_year is not None and work.year != expected_year:
        raise ValueError("Crossref publication year does not match the approved year.")
    return EvidenceSource(
        source_id=source_id,
        title=work.title,
        abstract="",
        year=work.year,
        doi=work.doi,
        url=f"https://doi.org/{work.doi}",
        verification_status=VerificationStatus.VERIFIED,
    )
