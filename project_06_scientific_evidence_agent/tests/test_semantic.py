"""Tests for the deterministic corpus-only latent-semantic retriever."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_agent.retrieval.semantic import (
    SemanticIndexError,
    build_lsa_index,
    load_lsa_index,
    write_lsa_index,
)


def _documents() -> dict[int, str]:
    return {
        1: "Immune response changes in disease.",
        2: "Bacterial infection and antibiotic treatment.",
        3: "Immune cells produce an inflammatory response.",
    }


def test_lsa_index_round_trip_preserves_deterministic_search(tmp_path: Path) -> None:
    index = build_lsa_index(
        _documents(),
        corpus_sha256="a" * 64,
        n_components=1,
        min_document_frequency=1,
    )
    index_path = tmp_path / "artifacts" / "semantic.joblib"

    write_lsa_index(index, index_path)
    restored = load_lsa_index(index_path)

    assert restored.summary_dict()["document_count"] == 3
    assert restored.search("immune response", k=2) == index.search("immune response", k=2)


def test_lsa_index_rejects_too_many_components() -> None:
    with pytest.raises(SemanticIndexError, match="n_components"):
        build_lsa_index(
            _documents(),
            corpus_sha256="a" * 64,
            n_components=3,
            min_document_frequency=1,
        )


def test_lsa_returns_no_result_for_an_out_of_vocabulary_query() -> None:
    index = build_lsa_index(
        _documents(),
        corpus_sha256="a" * 64,
        n_components=1,
        min_document_frequency=1,
    )

    assert index.search("unseen terminology", k=2) == ()
