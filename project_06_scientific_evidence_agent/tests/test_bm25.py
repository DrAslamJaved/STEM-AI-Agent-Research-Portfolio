"""Tests for the deterministic lexical retrieval baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_agent.retrieval.bm25 import (
    RetrievalIndexError,
    build_bm25_index,
    load_bm25_index,
    tokenize,
    write_bm25_index,
)


def test_tokenize_is_lowercase_and_deterministic() -> None:
    assert tokenize("RNA-seq, Version 2!") == ("rna", "seq", "version", "2")


def test_bm25_ranks_the_most_relevant_document_and_breaks_ties_by_id() -> None:
    index = build_bm25_index(
        {
            7: "A repeated enzyme inhibition mechanism.",
            2: "enzyme inhibition",
            9: "enzyme inhibition",
        }
    )

    hits = index.search("enzyme inhibition", k=3)

    assert [hit.doc_id for hit in hits] == [2, 9, 7]
    assert hits[0].score > 0


def test_bm25_round_trip_preserves_ranked_results(tmp_path: Path) -> None:
    index = build_bm25_index(
        {1: "immune response", 4: "immune response disease"},
        corpus_sha256="a" * 64,
    )
    index_path = tmp_path / "artifacts" / "index.json"

    write_bm25_index(index, index_path)
    restored = load_bm25_index(index_path)

    assert restored.corpus_sha256 == "a" * 64
    assert restored.search("immune disease", k=2) == index.search("immune disease", k=2)
    assert json.loads(index_path.read_text(encoding="utf-8"))["format"] == "evidence_agent_bm25_v1"


@pytest.mark.parametrize("k", [0, -1, True])
def test_bm25_rejects_invalid_retrieval_budget(k: int) -> None:
    index = build_bm25_index({1: "scientific evidence"})

    with pytest.raises(RetrievalIndexError, match="positive integer"):
        index.search("evidence", k=k)
