"""Tests for safe runtime SciFact loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_agent.retrieval.scifact import (
    RetrievalDataError,
    load_gold_evidence_documents,
    load_runtime_claims,
    load_scifact_corpus,
)
from tests.helpers import write_minimal_scifact_dataset


def test_runtime_claim_loader_discards_evaluator_only_fields(tmp_path: Path) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "scifact")

    claims = load_runtime_claims(dataset / "claims_train.jsonl")

    assert claims[0].claim_id == 1
    assert claims[0].text == "A supported claim."
    assert not hasattr(claims[0], "evidence")
    assert not hasattr(claims[0], "cited_doc_ids")


def test_corpus_and_gold_loaders_use_their_respective_boundaries(tmp_path: Path) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "scifact")

    corpus = load_scifact_corpus(dataset / "corpus.jsonl")
    gold = load_gold_evidence_documents(dataset / "claims_dev.jsonl")

    assert corpus[10].searchable_text == "Document one Sentence zero. Sentence one."
    assert gold == {2: frozenset({11})}


def test_runtime_claim_loader_rejects_blank_jsonl_line(tmp_path: Path) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "scifact")
    claim_path = dataset / "claims_train.jsonl"
    claim_path.write_text(claim_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RetrievalDataError, match="blank JSONL"):
        load_runtime_claims(claim_path)
