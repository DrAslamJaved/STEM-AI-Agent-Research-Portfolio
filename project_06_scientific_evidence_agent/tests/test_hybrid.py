"""Tests for fixed BM25 + LSA fusion and transparent candidate reranking."""

from __future__ import annotations

import pytest

from evidence_agent.data.schemas import Claim
from evidence_agent.retrieval.bm25 import build_bm25_index
from evidence_agent.retrieval.hybrid import (
    HybridRetrievalError,
    HybridRetriever,
    as_evaluation_predictions,
    retrieve_hybrid_claims,
)
from evidence_agent.retrieval.scifact import CorpusDocument
from evidence_agent.retrieval.semantic import build_lsa_index


def _corpus() -> dict[int, CorpusDocument]:
    return {
        1: CorpusDocument(
            doc_id=1,
            title="Immune response",
            abstract=("Inflammatory immune cells respond to disease.",),
        ),
        2: CorpusDocument(
            doc_id=2,
            title="Antibiotic treatment",
            abstract=("Antibiotics treat bacterial infection.",),
        ),
        3: CorpusDocument(
            doc_id=3,
            title="Immune mechanisms",
            abstract=("An immune response can be inflammatory.",),
        ),
    }


def _retriever() -> HybridRetriever:
    corpus = _corpus()
    texts = {doc_id: document.searchable_text for doc_id, document in corpus.items()}
    return HybridRetriever(
        bm25_index=build_bm25_index(texts, corpus_sha256="a" * 64),
        semantic_index=build_lsa_index(
            texts,
            corpus_sha256="a" * 64,
            n_components=1,
            min_document_frequency=1,
        ),
        corpus=corpus,
        candidate_k=3,
    )


def test_hybrid_retriever_is_deterministic_and_emits_diagnostics() -> None:
    retriever = _retriever()
    claim = Claim(claim_id=4, text="immune inflammatory response")

    first = retriever.search(claim, k=2)
    second = retriever.search(claim, k=2)

    assert first == second
    assert first[0].doc_id in {1, 3}
    assert first[0].rrf_score > 0
    assert first[0].semantic_rank is not None


def test_hybrid_predictions_convert_to_generic_evaluator_shape() -> None:
    predictions = retrieve_hybrid_claims(
        _retriever(),
        (Claim(claim_id=4, text="immune inflammatory response"),),
        top_k=2,
    )

    evaluation_predictions = as_evaluation_predictions(predictions)

    assert evaluation_predictions[0].claim_id == 4
    assert evaluation_predictions[0].hits[0].doc_id == predictions[0].hits[0].doc_id


def test_hybrid_retriever_rejects_mismatched_index_fingerprints() -> None:
    corpus = _corpus()
    texts = {doc_id: document.searchable_text for doc_id, document in corpus.items()}

    with pytest.raises(HybridRetrievalError, match="different corpus SHA-256"):
        HybridRetriever(
            bm25_index=build_bm25_index(texts, corpus_sha256="a" * 64),
            semantic_index=build_lsa_index(
                texts,
                corpus_sha256="b" * 64,
                n_components=1,
                min_document_frequency=1,
            ),
            corpus=corpus,
        )
