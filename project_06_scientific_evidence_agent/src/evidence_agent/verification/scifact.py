"""Train/evaluation-only SciFact adapters for the verification phase.

Unlike :mod:`evidence_agent.retrieval.scifact`, these functions may read
SciFact evidence and cited-document annotations.  They are deliberately kept
outside the runtime retrieval path: training uses only the train split and
evaluation loads annotations only after runtime decisions have frozen.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidence_agent.data.schemas import Citation, Verdict
from evidence_agent.retrieval.scifact import CorpusDocument, load_scifact_corpus
from evidence_agent.verification.models import (
    SentenceInput,
    SentenceTrainingExample,
    StanceInput,
    StanceTrainingExample,
)


class VerificationDataError(ValueError):
    """Raised when train/evaluation SciFact annotations are malformed."""


@dataclass(frozen=True, slots=True)
class VerificationTrainingData:
    """All train-split examples needed by the two lexical verification models."""

    stance_examples: tuple[StanceTrainingExample, ...]
    sentence_examples: tuple[SentenceTrainingExample, ...]
    claim_count: int

    def summary_dict(self) -> dict[str, object]:
        stance_counts = {label: 0 for label in Verdict}
        for example in self.stance_examples:
            stance_counts[example.label] += 1
        return {
            "claim_count": self.claim_count,
            "sentence_example_count": len(self.sentence_examples),
            "stance_example_count": len(self.stance_examples),
            "stance_label_counts": {str(label): count for label, count in stance_counts.items()},
        }


@dataclass(frozen=True, slots=True)
class GoldClaimAnnotation:
    """Evaluator-only claim verdict and complete gold citation rationale sets."""

    claim_id: int
    verdict: Verdict
    citations: tuple[Citation, ...]

    def __post_init__(self) -> None:
        if self.verdict is Verdict.NO_EVIDENCE and self.citations:
            raise VerificationDataError("NO_EVIDENCE gold annotations cannot contain citations.")
        if self.verdict is not Verdict.NO_EVIDENCE and not self.citations:
            raise VerificationDataError("Assertive gold annotations require citations.")


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise VerificationDataError(f"{path}:{line_number}: blank JSONL line.")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise VerificationDataError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}."
                    ) from error
                if not isinstance(record, Mapping):
                    raise VerificationDataError(
                        f"{path}:{line_number}: each JSONL record must be an object."
                    )
                yield line_number, record
    except OSError as error:
        raise VerificationDataError(f"Unable to read {path}: {error}") from error


def _require_non_negative_int(value: Any, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VerificationDataError(f"{source}: {name} must be a non-negative integer.")
    return value


def _require_text(value: Any, name: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerificationDataError(f"{source}: {name} must be non-empty text.")
    return value


def _document_text(document: CorpusDocument) -> str:
    return document.searchable_text


@dataclass(frozen=True, slots=True)
class _ParsedClaim:
    claim_id: int
    claim_text: str
    cited_doc_ids: tuple[int, ...]
    evidence_by_doc: Mapping[int, tuple[Citation, ...]]


def _parse_claim(
    record: Mapping[str, Any],
    corpus: Mapping[int, CorpusDocument],
    source: str,
) -> _ParsedClaim:
    claim_id = _require_non_negative_int(record.get("id"), "id", source)
    claim_text = _require_text(record.get("claim"), "claim", source)
    raw_cited_doc_ids = record.get("cited_doc_ids", [])
    if not isinstance(raw_cited_doc_ids, list):
        raise VerificationDataError(f"{source}: cited_doc_ids must be a list.")
    cited_doc_ids = tuple(
        dict.fromkeys(
            _require_non_negative_int(doc_id, "cited_doc_id", source)
            for doc_id in raw_cited_doc_ids
        )
    )
    if any(doc_id not in corpus for doc_id in cited_doc_ids):
        raise VerificationDataError(f"{source}: cited_doc_ids reference a missing corpus document.")

    raw_evidence = record.get("evidence", {})
    if not isinstance(raw_evidence, Mapping):
        raise VerificationDataError(f"{source}: evidence must be an object.")
    evidence_by_doc: dict[int, tuple[Citation, ...]] = {}
    for raw_doc_id, raw_rationales in raw_evidence.items():
        try:
            doc_id = int(raw_doc_id)
        except (TypeError, ValueError) as error:
            raise VerificationDataError(
                f"{source}: invalid evidence document ID {raw_doc_id!r}."
            ) from error
        if str(doc_id) != str(raw_doc_id) or doc_id not in corpus:
            raise VerificationDataError(
                f"{source}: evidence document {raw_doc_id!r} is absent from the corpus."
            )
        if not isinstance(raw_rationales, list) or not raw_rationales:
            raise VerificationDataError(f"{source}: evidence[{raw_doc_id!r}] must be non-empty.")
        rationales: list[Citation] = []
        for raw_rationale in raw_rationales:
            if not isinstance(raw_rationale, Mapping):
                raise VerificationDataError(f"{source}: rationale must be an object.")
            try:
                stance = Verdict(raw_rationale.get("label"))
            except ValueError as error:
                raise VerificationDataError(
                    f"{source}: rationale label must be SUPPORT or CONTRADICT."
                ) from error
            if stance is Verdict.NO_EVIDENCE:
                raise VerificationDataError(f"{source}: NO_EVIDENCE cannot be a rationale label.")
            sentence_ids = raw_rationale.get("sentences")
            if not isinstance(sentence_ids, list) or not sentence_ids:
                raise VerificationDataError(f"{source}: rationale sentences must be non-empty.")
            normalised_sentence_ids = tuple(
                sorted(
                    {
                        _require_non_negative_int(sentence_id, "rationale sentence id", source)
                        for sentence_id in sentence_ids
                    }
                )
            )
            if len(normalised_sentence_ids) != len(sentence_ids):
                raise VerificationDataError(f"{source}: rationale sentence IDs must be unique.")
            if normalised_sentence_ids[-1] >= len(corpus[doc_id].abstract):
                raise VerificationDataError(
                    f"{source}: rationale sentence ID is outside document {doc_id}."
                )
            rationales.append(Citation(doc_id=doc_id, sentence_ids=normalised_sentence_ids, stance=stance))
        labels = {citation.stance for citation in rationales}
        if len(labels) != 1:
            raise VerificationDataError(
                f"{source}: one evidence document cannot have conflicting rationale labels."
            )
        evidence_by_doc[doc_id] = tuple(
            sorted(rationales, key=lambda citation: (citation.stance, citation.sentence_ids))
        )

    if set(evidence_by_doc) - set(cited_doc_ids):
        raise VerificationDataError(
            f"{source}: every evidence document must also be listed in cited_doc_ids."
        )
    return _ParsedClaim(
        claim_id=claim_id,
        claim_text=claim_text,
        cited_doc_ids=cited_doc_ids,
        evidence_by_doc=evidence_by_doc,
    )


def _normalise_claim_ids(claim_ids: Collection[int] | None) -> frozenset[int] | None:
    if claim_ids is None:
        return None
    normalised: set[int] = set()
    for claim_id in claim_ids:
        normalised.add(_require_non_negative_int(claim_id, "claim_id", "claim_ids"))
    if not normalised:
        raise VerificationDataError("claim_ids must contain at least one claim ID.")
    return frozenset(normalised)


def _iter_parsed_claims(
    claims_path: Path,
    corpus: Mapping[int, CorpusDocument],
    *,
    claim_ids: Collection[int] | None = None,
) -> Iterator[_ParsedClaim]:
    allowed_claim_ids = _normalise_claim_ids(claim_ids)
    seen_claim_ids: set[int] = set()
    for line_number, record in _iter_jsonl(claims_path):
        source = f"{claims_path}:{line_number}"
        raw_claim_id = _require_non_negative_int(record.get("id"), "id", source)
        if allowed_claim_ids is not None and raw_claim_id not in allowed_claim_ids:
            continue
        parsed = _parse_claim(record, corpus, source)
        if parsed.claim_id in seen_claim_ids:
            raise VerificationDataError(f"{claims_path}:{line_number}: duplicate claim ID.")
        seen_claim_ids.add(parsed.claim_id)
        yield parsed


def load_verification_training_data(
    train_claims_path: Path,
    corpus_path: Path,
    *,
    claim_ids: Collection[int] | None = None,
) -> VerificationTrainingData:
    """Create labelled examples strictly from the SciFact training split.

    Evidence documents receive their supplied SUPPORT or CONTRADICT label.
    Cited-but-not-evidence documents are NO_EVIDENCE examples.  Sentence labels
    are positive exactly when a sentence belongs to a gold rationale set.
    """
    corpus = load_scifact_corpus(corpus_path)
    stance_examples: list[StanceTrainingExample] = []
    sentence_examples: list[SentenceTrainingExample] = []
    claim_count = 0
    for parsed in _iter_parsed_claims(
        train_claims_path,
        corpus,
        claim_ids=claim_ids,
    ):
        claim_count += 1
        for doc_id in parsed.cited_doc_ids:
            document = corpus[doc_id]
            citations = parsed.evidence_by_doc.get(doc_id, ())
            label = citations[0].stance if citations else Verdict.NO_EVIDENCE
            stance_input = StanceInput(
                claim_id=parsed.claim_id,
                doc_id=doc_id,
                claim_text=parsed.claim_text,
                document_text=_document_text(document),
            )
            stance_examples.append(StanceTrainingExample(input=stance_input, label=label))
            positive_sentence_ids = {
                sentence_id for citation in citations for sentence_id in citation.sentence_ids
            }
            for sentence_id, sentence_text in enumerate(document.abstract):
                sentence_examples.append(
                    SentenceTrainingExample(
                        input=SentenceInput(
                            claim_id=parsed.claim_id,
                            doc_id=doc_id,
                            sentence_id=sentence_id,
                            claim_text=parsed.claim_text,
                            sentence_text=sentence_text,
                        ),
                        is_evidence=sentence_id in positive_sentence_ids,
                    )
                )
    if not stance_examples or not sentence_examples:
        raise VerificationDataError(f"{train_claims_path}: no training examples were generated.")
    return VerificationTrainingData(
        stance_examples=tuple(stance_examples),
        sentence_examples=tuple(sentence_examples),
        claim_count=claim_count,
    )


def load_stance_benchmark_inputs(claims_path: Path, corpus_path: Path) -> tuple[StanceInput, ...]:
    """Load cited-document benchmark inputs without reading evidence labels."""
    corpus = load_scifact_corpus(corpus_path)
    inputs: list[StanceInput] = []
    seen_claim_ids: set[int] = set()
    for line_number, record in _iter_jsonl(claims_path):
        source = f"{claims_path}:{line_number}"
        claim_id = _require_non_negative_int(record.get("id"), "id", source)
        if claim_id in seen_claim_ids:
            raise VerificationDataError(f"{source}: duplicate claim ID.")
        seen_claim_ids.add(claim_id)
        claim_text = _require_text(record.get("claim"), "claim", source)
        raw_cited_doc_ids = record.get("cited_doc_ids", [])
        if not isinstance(raw_cited_doc_ids, list) or not raw_cited_doc_ids:
            raise VerificationDataError(f"{source}: cited_doc_ids must be a non-empty list.")
        cited_doc_ids = tuple(
            dict.fromkeys(
                _require_non_negative_int(doc_id, "cited_doc_id", source)
                for doc_id in raw_cited_doc_ids
            )
        )
        for doc_id in cited_doc_ids:
            if doc_id not in corpus:
                raise VerificationDataError(f"{source}: cited_doc_id is absent from corpus.")
            inputs.append(
                StanceInput(
                    claim_id=claim_id,
                    doc_id=doc_id,
                    claim_text=claim_text,
                    document_text=_document_text(corpus[doc_id]),
                )
            )
    if not inputs:
        raise VerificationDataError(f"{claims_path}: no cited-document benchmark pairs.")
    return tuple(inputs)


def load_stance_benchmark_labels(claims_path: Path, corpus_path: Path) -> tuple[Verdict, ...]:
    """Load evaluator-only labels only after benchmark model outputs freeze."""
    corpus = load_scifact_corpus(corpus_path)
    labels: list[Verdict] = []
    for parsed in _iter_parsed_claims(claims_path, corpus):
        for doc_id in parsed.cited_doc_ids:
            citations = parsed.evidence_by_doc.get(doc_id, ())
            labels.append(citations[0].stance if citations else Verdict.NO_EVIDENCE)
    if not labels:
        raise VerificationDataError(f"{claims_path}: no cited-document benchmark labels.")
    return tuple(labels)


def load_stance_benchmark(
    claims_path: Path, corpus_path: Path
) -> tuple[tuple[StanceInput, ...], tuple[Verdict, ...]]:
    """Convenience loader for test code; production evaluation uses separated calls."""
    return (
        load_stance_benchmark_inputs(claims_path, corpus_path),
        load_stance_benchmark_labels(claims_path, corpus_path),
    )


def load_gold_claim_annotations(
    claims_path: Path,
    corpus_path: Path,
    *,
    claim_ids: Collection[int] | None = None,
) -> dict[int, GoldClaimAnnotation]:
    """Load evaluator-only claim labels and exact SciFact rationale sets."""
    corpus = load_scifact_corpus(corpus_path)
    annotations: dict[int, GoldClaimAnnotation] = {}
    for parsed in _iter_parsed_claims(claims_path, corpus, claim_ids=claim_ids):
        citations = tuple(
            citation
            for doc_id in sorted(parsed.evidence_by_doc)
            for citation in parsed.evidence_by_doc[doc_id]
        )
        labels = {citation.stance for citation in citations}
        if len(labels) > 1:
            raise VerificationDataError(
                f"{claims_path}: claim {parsed.claim_id} has conflicting evidence labels."
            )
        verdict = next(iter(labels), Verdict.NO_EVIDENCE)
        annotations[parsed.claim_id] = GoldClaimAnnotation(
            claim_id=parsed.claim_id,
            verdict=verdict,
            citations=citations,
        )
    if not annotations:
        raise VerificationDataError(f"{claims_path}: no gold claim annotations.")
    return annotations


def citations_to_sentence_keys(
    citations: Sequence[Citation], claim_id: int
) -> frozenset[tuple[int, int, int, Verdict]]:
    """Expand citations to comparable ``(claim, doc, sentence, stance)`` keys."""
    return frozenset(
        (claim_id, citation.doc_id, sentence_id, citation.stance)
        for citation in citations
        for sentence_id in citation.sentence_ids
    )
