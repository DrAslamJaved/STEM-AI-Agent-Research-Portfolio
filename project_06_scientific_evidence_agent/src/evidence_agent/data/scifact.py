"""SciFact structural validation, isolated from runtime retrieval code."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CLAIM_SPLITS = ("train", "dev", "test")
ALLOWED_EVIDENCE_LABELS = frozenset({"SUPPORT", "CONTRADICT"})
REQUIRED_DATASET_FILES = frozenset(
    {"corpus.jsonl", *(f"claims_{split}.jsonl" for split in CLAIM_SPLITS)}
)


class DatasetValidationError(ValueError):
    """Raised when SciFact fails its documented structural contract."""


@dataclass(frozen=True, slots=True)
class SciFactValidationSummary:
    """Portable summary emitted after full dataset structural validation."""

    dataset_root: str
    corpus_documents: int
    corpus_sentences: int
    claims_by_split: dict[str, int]
    evidence_documents: int
    evidence_sets: int
    evidence_sentences: int
    evidence_labels: dict[str, int]
    cited_documents_not_in_corpus: int
    cross_validation_folds: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _source(path: Path, line_number: int | None = None) -> str:
    return f"{path}:{line_number}" if line_number is not None else str(path)


def _require_int(value: Any, name: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetValidationError(f"{source}: {name} must be a non-negative integer.")
    return value


def _require_text(value: Any, name: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{source}: {name} must be a non-empty string.")
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetValidationError(
                        f"{_source(path, line_number)}: blank JSONL lines are not allowed."
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetValidationError(
                        f"{_source(path, line_number)}: invalid JSON: {error.msg}."
                    ) from error
                if not isinstance(record, Mapping):
                    raise DatasetValidationError(
                        f"{_source(path, line_number)}: each JSONL record must be an object."
                    )
                yield line_number, record
    except OSError as error:
        raise DatasetValidationError(f"Unable to read {path}: {error}") from error


def resolve_scifact_dataset_root(data_dir: Path) -> Path:
    """Find exactly one directory containing the official SciFact file layout."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise DatasetValidationError(f"SciFact data directory does not exist: {data_dir}.")

    candidates = [data_dir]
    candidates.extend(path.parent for path in data_dir.rglob("corpus.jsonl"))
    matches = [
        candidate.resolve()
        for candidate in candidates
        if candidate.is_dir() and REQUIRED_DATASET_FILES.issubset({path.name for path in candidate.iterdir()})
    ]
    unique_matches = list(dict.fromkeys(matches))
    if not unique_matches:
        expected = ", ".join(sorted(REQUIRED_DATASET_FILES))
        raise DatasetValidationError(
            f"No SciFact dataset root found under {data_dir}; expected {expected}."
        )
    if len(unique_matches) > 1:
        raise DatasetValidationError(
            f"Ambiguous SciFact dataset roots under {data_dir}: {unique_matches}."
        )
    return unique_matches[0]


def _validate_corpus(corpus_path: Path) -> dict[int, int]:
    sentence_counts: dict[int, int] = {}
    for line_number, record in _iter_jsonl(corpus_path):
        source = _source(corpus_path, line_number)
        doc_id = _require_int(record.get("doc_id"), "doc_id", source)
        if doc_id in sentence_counts:
            raise DatasetValidationError(f"{source}: duplicate corpus doc_id {doc_id}.")
        _require_text(record.get("title"), "title", source)
        abstract = record.get("abstract")
        if not isinstance(abstract, list) or not abstract:
            raise DatasetValidationError(
                f"{source}: abstract must be a non-empty list of sentences."
            )
        for sentence_number, sentence in enumerate(abstract):
            _require_text(sentence, f"abstract[{sentence_number}]", source)
        if not isinstance(record.get("structured"), bool):
            raise DatasetValidationError(f"{source}: structured must be boolean.")
        sentence_counts[doc_id] = len(abstract)
    if not sentence_counts:
        raise DatasetValidationError(f"{corpus_path}: corpus is empty.")
    return sentence_counts


@dataclass(slots=True)
class _ClaimValidationStats:
    count: int = 0
    evidence_documents: int = 0
    evidence_sets: int = 0
    evidence_sentences: int = 0
    cited_documents_not_in_corpus: int = 0
    labels: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = Counter()


def _validate_claim_file(
    claim_path: Path,
    corpus_sentence_counts: Mapping[int, int],
    seen_claim_ids: set[int],
    *,
    allowed_claim_ids: set[int] | None = None,
) -> _ClaimValidationStats:
    stats = _ClaimValidationStats()
    for line_number, record in _iter_jsonl(claim_path):
        source = _source(claim_path, line_number)
        claim_id = _require_int(record.get("id"), "id", source)
        if claim_id in seen_claim_ids:
            raise DatasetValidationError(f"{source}: duplicate claim id {claim_id}.")
        if allowed_claim_ids is not None and claim_id not in allowed_claim_ids:
            raise DatasetValidationError(
                f"{source}: cross-validation claim id {claim_id} is absent from train/dev."
            )
        seen_claim_ids.add(claim_id)
        _require_text(record.get("claim"), "claim", source)

        cited_doc_ids = record.get("cited_doc_ids", [])
        if not isinstance(cited_doc_ids, list):
            raise DatasetValidationError(f"{source}: cited_doc_ids must be a list.")
        for cited_doc_id in cited_doc_ids:
            cited_doc_id = _require_int(cited_doc_id, "cited_doc_id", source)
            if cited_doc_id not in corpus_sentence_counts:
                stats.cited_documents_not_in_corpus += 1

        evidence = record.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise DatasetValidationError(f"{source}: evidence must be an object.")
        for raw_doc_id, rationale_sets in evidence.items():
            try:
                doc_id = int(raw_doc_id)
            except (TypeError, ValueError) as error:
                raise DatasetValidationError(
                    f"{source}: evidence document id {raw_doc_id!r} is invalid."
                ) from error
            if str(doc_id) != str(raw_doc_id) or doc_id not in corpus_sentence_counts:
                raise DatasetValidationError(
                    f"{source}: evidence document id {raw_doc_id!r} is absent from corpus."
                )
            if not isinstance(rationale_sets, list) or not rationale_sets:
                raise DatasetValidationError(
                    f"{source}: evidence[{raw_doc_id!r}] must contain rationale sets."
                )
            stats.evidence_documents += 1
            for rationale in rationale_sets:
                if not isinstance(rationale, Mapping):
                    raise DatasetValidationError(
                        f"{source}: rationale must be an object."
                    )
                label = rationale.get("label")
                if label not in ALLOWED_EVIDENCE_LABELS:
                    raise DatasetValidationError(
                        f"{source}: invalid evidence label {label!r}."
                    )
                sentence_ids = rationale.get("sentences")
                if not isinstance(sentence_ids, list) or not sentence_ids:
                    raise DatasetValidationError(
                        f"{source}: rationale sentences must be a non-empty list."
                    )
                if len(sentence_ids) != len(set(sentence_ids)):
                    raise DatasetValidationError(
                        f"{source}: rationale sentence ids must be unique."
                    )
                for sentence_id in sentence_ids:
                    sentence_id = _require_int(sentence_id, "rationale sentence id", source)
                    if sentence_id >= corpus_sentence_counts[doc_id]:
                        raise DatasetValidationError(
                            f"{source}: sentence id {sentence_id} is outside document {doc_id}."
                        )
                stats.evidence_sets += 1
                stats.evidence_sentences += len(sentence_ids)
                stats.labels[label] += 1
        stats.count += 1
    if stats.count == 0:
        raise DatasetValidationError(f"{claim_path}: claim split is empty.")
    return stats


def _validate_cross_validation_layout(
    root: Path,
    corpus_sentence_counts: Mapping[int, int],
    allowed_claim_ids: set[int],
    require_cross_validation: bool,
) -> int:
    cross_validation_dir = root / "cross_validation"
    if not cross_validation_dir.exists():
        if require_cross_validation:
            raise DatasetValidationError(
                f"{cross_validation_dir}: expected five-fold cross-validation data."
            )
        return 0

    expected_folds = [cross_validation_dir / f"fold_{fold}" for fold in range(1, 6)]
    missing = [fold for fold in expected_folds if not fold.is_dir()]
    if missing:
        raise DatasetValidationError(f"Missing cross-validation fold(s): {missing}.")

    for fold_number, fold_dir in enumerate(expected_folds, start=1):
        seen_fold_ids: set[int] = set()
        for split in ("train", "dev"):
            claim_path = fold_dir / f"claims_{split}_{fold_number}.jsonl"
            if not claim_path.is_file():
                raise DatasetValidationError(
                    f"{fold_dir}: missing expected file {claim_path.name}."
                )
            _validate_claim_file(
                claim_path,
                corpus_sentence_counts,
                seen_fold_ids,
                allowed_claim_ids=allowed_claim_ids,
            )
    return len(expected_folds)


def validate_scifact_dataset(
    data_dir: Path, *, require_cross_validation: bool = True
) -> SciFactValidationSummary:
    """Validate all public SciFact JSONL files and return a stable summary."""
    root = resolve_scifact_dataset_root(data_dir)
    corpus_sentence_counts = _validate_corpus(root / "corpus.jsonl")

    claims_by_split: dict[str, int] = {}
    total_evidence_documents = 0
    total_evidence_sets = 0
    total_evidence_sentences = 0
    total_missing_cited_documents = 0
    labels: Counter[str] = Counter()
    seen_primary_claim_ids: set[int] = set()
    labelled_claim_ids: set[int] = set()

    for split in CLAIM_SPLITS:
        stats = _validate_claim_file(
            root / f"claims_{split}.jsonl",
            corpus_sentence_counts,
            seen_primary_claim_ids,
        )
        claims_by_split[split] = stats.count
        total_evidence_documents += stats.evidence_documents
        total_evidence_sets += stats.evidence_sets
        total_evidence_sentences += stats.evidence_sentences
        total_missing_cited_documents += stats.cited_documents_not_in_corpus
        labels.update(stats.labels)
        if split != "test":
            labelled_claim_ids.update(seen_primary_claim_ids)

    cross_validation_folds = _validate_cross_validation_layout(
        root,
        corpus_sentence_counts,
        labelled_claim_ids,
        require_cross_validation,
    )

    return SciFactValidationSummary(
        dataset_root=str(root),
        corpus_documents=len(corpus_sentence_counts),
        corpus_sentences=sum(corpus_sentence_counts.values()),
        claims_by_split=claims_by_split,
        evidence_documents=total_evidence_documents,
        evidence_sets=total_evidence_sets,
        evidence_sentences=total_evidence_sentences,
        evidence_labels=dict(sorted(labels.items())),
        cited_documents_not_in_corpus=total_missing_cited_documents,
        cross_validation_folds=cross_validation_folds,
    )


def write_validation_report(summary: SciFactValidationSummary, path: Path) -> None:
    """Write a deterministic JSON validation artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
