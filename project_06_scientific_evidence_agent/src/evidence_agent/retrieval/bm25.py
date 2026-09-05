"""A deterministic, dependency-free BM25 retrieval baseline."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INDEX_FORMAT = "evidence_agent_bm25_v1"
TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


class RetrievalIndexError(ValueError):
    """Raised when a BM25 index or its inputs are malformed."""


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """One ranked document returned by the lexical retriever."""

    doc_id: int
    score: float

    def as_dict(self) -> dict[str, object]:
        return {"doc_id": self.doc_id, "score": self.score}


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text deterministically without a model or external resource."""
    if not isinstance(text, str):
        raise RetrievalIndexError("BM25 input text must be a string.")
    return tuple(TOKEN_PATTERN.findall(text.lower()))


def _validate_parameters(k1: float, b: float) -> None:
    if not isinstance(k1, (int, float)) or not math.isfinite(k1) or k1 <= 0:
        raise RetrievalIndexError("BM25 k1 must be a finite positive number.")
    if not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b <= 1:
        raise RetrievalIndexError("BM25 b must be a finite number in [0, 1].")


@dataclass(frozen=True, slots=True)
class BM25Index:
    """A serializable inverted BM25 index with deterministic tie breaking."""

    document_lengths: dict[int, int]
    postings: dict[str, tuple[tuple[int, int], ...]]
    average_document_length: float
    k1: float = 1.2
    b: float = 0.75
    corpus_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_parameters(self.k1, self.b)
        if not self.document_lengths:
            raise RetrievalIndexError("BM25 index must contain at least one document.")
        if self.average_document_length <= 0:
            raise RetrievalIndexError("BM25 average document length must be positive.")

    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self.document_lengths)

    @property
    def vocabulary_size(self) -> int:
        """Return the number of distinct indexed tokens."""
        return len(self.postings)

    def search(self, query: str, k: int) -> tuple[RetrievalHit, ...]:
        """Return up to *k* positive-score results, ordered by score then ID."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise RetrievalIndexError("Retrieval k must be a positive integer.")

        query_terms = sorted(set(tokenize(query)))
        scores: defaultdict[int, float] = defaultdict(float)
        document_count = self.document_count

        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for doc_id, term_frequency in postings:
                document_length = self.document_lengths[doc_id]
                normalizer = self.k1 * (
                    1 - self.b + self.b * document_length / self.average_document_length
                )
                scores[doc_id] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1) / (term_frequency + normalizer)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return tuple(RetrievalHit(doc_id=doc_id, score=score) for doc_id, score in ranked)

    def as_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable index representation."""
        return {
            "format": INDEX_FORMAT,
            "parameters": {"b": self.b, "k1": self.k1},
            "corpus_sha256": self.corpus_sha256,
            "average_document_length": self.average_document_length,
            "document_lengths": {
                str(doc_id): length
                for doc_id, length in sorted(self.document_lengths.items())
            },
            "postings": {
                term: [[doc_id, frequency] for doc_id, frequency in postings]
                for term, postings in sorted(self.postings.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BM25Index":
        """Restore an index written by :meth:`as_dict`."""
        if payload.get("format") != INDEX_FORMAT:
            raise RetrievalIndexError(
                f"Unsupported BM25 index format: {payload.get('format')!r}."
            )
        try:
            parameters = payload["parameters"]
            document_lengths = {
                int(doc_id): int(length)
                for doc_id, length in payload["document_lengths"].items()
            }
            postings = {
                str(term): tuple((int(doc_id), int(frequency)) for doc_id, frequency in rows)
                for term, rows in payload["postings"].items()
            }
            average_document_length = float(payload["average_document_length"])
            k1 = float(parameters["k1"])
            b = float(parameters["b"])
            corpus_sha256 = payload.get("corpus_sha256")
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RetrievalIndexError("Malformed BM25 index payload.") from error

        if any(doc_id < 0 or length <= 0 for doc_id, length in document_lengths.items()):
            raise RetrievalIndexError("BM25 document IDs and lengths must be positive.")
        if any(
            doc_id not in document_lengths or frequency <= 0
            for rows in postings.values()
            for doc_id, frequency in rows
        ):
            raise RetrievalIndexError("BM25 postings reference an invalid document or frequency.")
        if corpus_sha256 is not None and not isinstance(corpus_sha256, str):
            raise RetrievalIndexError("BM25 corpus_sha256 must be a string or null.")

        return cls(
            document_lengths=document_lengths,
            postings=postings,
            average_document_length=average_document_length,
            k1=k1,
            b=b,
            corpus_sha256=corpus_sha256,
        )


def build_bm25_index(
    documents: dict[int, str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
    corpus_sha256: str | None = None,
) -> BM25Index:
    """Build a BM25 index from public corpus documents only."""
    _validate_parameters(k1, b)
    if not documents:
        raise RetrievalIndexError("Cannot build a BM25 index from an empty corpus.")

    document_lengths: dict[int, int] = {}
    postings: dict[str, list[tuple[int, int]]] = {}
    for doc_id, text in sorted(documents.items()):
        if isinstance(doc_id, bool) or not isinstance(doc_id, int) or doc_id < 0:
            raise RetrievalIndexError("BM25 document IDs must be non-negative integers.")
        tokens = tokenize(text)
        if not tokens:
            raise RetrievalIndexError(f"BM25 document {doc_id} has no indexable tokens.")
        document_lengths[doc_id] = len(tokens)
        for term, frequency in sorted(Counter(tokens).items()):
            postings.setdefault(term, []).append((doc_id, frequency))

    average_document_length = sum(document_lengths.values()) / len(document_lengths)
    return BM25Index(
        document_lengths=document_lengths,
        postings={term: tuple(rows) for term, rows in postings.items()},
        average_document_length=average_document_length,
        k1=float(k1),
        b=float(b),
        corpus_sha256=corpus_sha256,
    )


def write_bm25_index(index: BM25Index, path: Path) -> None:
    """Write a stable BM25 artifact, creating its parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(index.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_bm25_index(path: Path) -> BM25Index:
    """Read a BM25 artifact and validate its schema."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalIndexError(f"Unable to read BM25 index {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RetrievalIndexError("BM25 index root must be a JSON object.")
    return BM25Index.from_dict(payload)
