"""Tests for frozen-prediction Recall@k evaluation."""

from __future__ import annotations

import pytest

from evidence_agent.data.schemas import Claim
from evidence_agent.evaluation.retrieval import (
    RetrievalEvaluationError,
    RetrievalPrediction,
    evaluate_retrieval_predictions,
    retrieve_claims,
)
from evidence_agent.retrieval.bm25 import RetrievalHit, build_bm25_index


def test_retrieval_evaluation_reports_claim_and_evidence_recall() -> None:
    predictions = (
        RetrievalPrediction(
            claim_id=1,
            hits=(RetrievalHit(doc_id=10, score=2.0), RetrievalHit(doc_id=12, score=1.0)),
        ),
        RetrievalPrediction(
            claim_id=2,
            hits=(RetrievalHit(doc_id=99, score=2.0), RetrievalHit(doc_id=11, score=1.0)),
        ),
    )

    result = evaluate_retrieval_predictions(
        predictions,
        {1: frozenset({10, 12}), 2: frozenset({11})},
        cutoffs=[2, 1],
    )

    assert result.cutoffs == (1, 2)
    assert result.claim_recall_at_k == {1: 0.5, 2: 1.0}
    assert result.evidence_document_recall_at_k == {1: 1 / 3, 2: 1.0}
    assert result.mean_reciprocal_rank == 0.75


def test_retrieval_evaluation_excludes_claims_without_gold_evidence() -> None:
    predictions = (
        RetrievalPrediction(claim_id=1, hits=(RetrievalHit(doc_id=10, score=1.0),)),
        RetrievalPrediction(claim_id=2, hits=()),
    )

    result = evaluate_retrieval_predictions(
        predictions,
        {1: frozenset({10}), 2: frozenset()},
        cutoffs=[1],
    )

    assert result.claim_count == 2
    assert result.gold_bearing_claim_count == 1
    assert result.claims_without_gold_evidence == 1


def test_retrieval_evaluation_rejects_gold_claim_without_prediction() -> None:
    with pytest.raises(RetrievalEvaluationError, match="Missing retrieval"):
        evaluate_retrieval_predictions((), {9: frozenset({3})}, cutoffs=[1])


def test_retrieve_claims_accepts_only_safe_claim_objects() -> None:
    index = build_bm25_index({10: "supported biological mechanism"})

    predictions = retrieve_claims(
        index,
        (Claim(claim_id=1, text="biological mechanism"),),
        top_k=1,
    )

    assert predictions[0].claim_id == 1
    assert predictions[0].hits[0].doc_id == 10
