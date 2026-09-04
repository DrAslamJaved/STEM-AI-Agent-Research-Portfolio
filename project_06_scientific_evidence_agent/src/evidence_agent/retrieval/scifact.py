"""Safe SciFact corpus and claim loaders for runtime retrieval."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_agent.contracts import assert_runtime_payload_is_safe, runtime_claim_from_scifact
from evidence_agent.data.schemas import Claim


class RetrievalDataError(ValueError):
    """Raised when retrieval-facing SciFact data do not meet the runtime contract."""


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """Public corpus content used by the lexical retriever."""

    doc_id: int
    title: str
    abstract: tuple[str, ...]

    @property
    def searchable_text(self) -> str:
        """Return the deterministic title-and-abstract representation."""
        return " ".join((self.title, *self.abstract))


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RetrievalDataError(f"{path}:{line_number}: blank JSONL line.")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RetrievalDataError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}."
                    ) from error
                if not isinstance(record, Mapping):
                    raise RetrievalDataError(
                        f"{path}:{line_number}: each JSONL record must be an object."
                    )
                yield line_number, record
    except OSError as error:
        raise RetrievalDataError(f"Unable to read {path}: {error}") from error


def _require_non_negative_int(value: Any, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalDataError(f"{source}: {name} must be a non-negative integer.")
    return value


def _require_text(value: Any, name: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalDataError(f"{source}: {name} must be non-empty text.")
    return value


def load_scifact_corpus(corpus_path: Path) -> dict[int, CorpusDocument]:
    """Load public SciFact documents with no claim annotations involved."""
    corpus: dict[int, CorpusDocument] = {}
    for line_number, raw_document in _iter_jsonl(corpus_path):
        source = f"{corpus_path}:{line_number}"
        doc_id = _require_non_negative_int(raw_document.get("doc_id"), "doc_id", source)
        if doc_id in corpus:
            raise RetrievalDataError(f"{source}: duplicate corpus document ID {doc_id}.")
        title = _require_text(raw_document.get("title"), "title", source)
        raw_abstract = raw_document.get("abstract")
        if not isinstance(raw_abstract, list) or not raw_abstract:
            raise RetrievalDataError(f"{source}: abstract must be a non-empty sentence list.")
        abstract = tuple(
            _require_text(sentence, f"abstract[{position}]", source)
            for position, sentence in enumerate(raw_abstract)
        )
        corpus[doc_id] = CorpusDocument(doc_id=doc_id, title=title, abstract=abstract)
    if not corpus:
        raise RetrievalDataError(f"{corpus_path}: corpus is empty.")
    return corpus


def load_runtime_claims(claims_path: Path) -> tuple[Claim, ...]:
    """Load only safe ``Claim(id, text)`` objects for a runtime retrieval call."""
    claims: list[Claim] = []
    seen_claim_ids: set[int] = set()
    for line_number, raw_claim in _iter_jsonl(claims_path):
        source = f"{claims_path}:{line_number}"
        runtime_payload = {"id": raw_claim.get("id"), "claim": raw_claim.get("claim")}
        assert_runtime_payload_is_safe(runtime_payload)
        try:
            claim = runtime_claim_from_scifact(runtime_payload)
        except ValueError as error:
            raise RetrievalDataError(f"{source}: invalid runtime claim: {error}") from error
        if claim.claim_id in seen_claim_ids:
            raise RetrievalDataError(f"{source}: duplicate claim ID {claim.claim_id}.")
        seen_claim_ids.add(claim.claim_id)
        claims.append(claim)
    if not claims:
        raise RetrievalDataError(f"{claims_path}: claim file is empty.")
    return tuple(claims)


def load_gold_evidence_documents(claims_path: Path) -> dict[int, frozenset[int]]:
    """Load evaluator-only gold evidence document IDs after retrieval completes."""
    gold_documents: dict[int, frozenset[int]] = {}
    for line_number, raw_claim in _iter_jsonl(claims_path):
        source = f"{claims_path}:{line_number}"
        claim_id = _require_non_negative_int(raw_claim.get("id"), "id", source)
        if claim_id in gold_documents:
            raise RetrievalDataError(f"{source}: duplicate claim ID {claim_id}.")
        raw_evidence = raw_claim.get("evidence", {})
        if not isinstance(raw_evidence, Mapping):
            raise RetrievalDataError(f"{source}: evidence must be an object for evaluation.")
        document_ids: set[int] = set()
        for raw_doc_id in raw_evidence:
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError) as error:
                raise RetrievalDataError(
                    f"{source}: evidence document ID {raw_doc_id!r} is invalid."
                ) from error
            if str(doc_id) != str(raw_doc_id) or doc_id < 0:
                raise RetrievalDataError(
                    f"{source}: evidence document ID {raw_doc_id!r} is invalid."
                )
            document_ids.add(doc_id)
        gold_documents[claim_id] = frozenset(document_ids)
    if not gold_documents:
        raise RetrievalDataError(f"{claims_path}: claim file is empty.")
    return gold_documents
