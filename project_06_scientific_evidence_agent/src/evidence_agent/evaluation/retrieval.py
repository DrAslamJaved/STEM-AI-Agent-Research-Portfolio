"""Leakage-safe evidence-document retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evidence_agent.data.schemas import Claim
from evidence_agent.retrieval.bm25 import BM25Index, RetrievalHit


class RetrievalEvaluationError(ValueError):
    """Raised when predictions and evaluator-only gold data are incompatible."""


@dataclass(frozen=True, slots=True)
class RetrievalPrediction:
    """Frozen runtime output for one claim, without evaluator annotations."""

    claim_id: int
    hits: tuple[RetrievalHit, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "retrieved_documents": [hit.as_dict() for hit in self.hits],
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    """Claim-level and document-level Recall@k for frozen predictions."""

    claim_count: int
    gold_bearing_claim_count: int
    claims_without_gold_evidence: int
    cutoffs: tuple[int, ...]
    claim_recall_at_k: dict[int, float]
    evidence_document_recall_at_k: dict[int, float]
    mean_reciprocal_rank: float
    predictions: tuple[RetrievalPrediction, ...]

    def summary_dict(self) -> dict[str, object]:
        """Return metrics only, suitable for terminal output and comparison tables."""
        return {
            "claim_count": self.claim_count,
            "gold_bearing_claim_count": self.gold_bearing_claim_count,
            "claims_without_gold_evidence": self.claims_without_gold_evidence,
            "cutoffs": list(self.cutoffs),
            "claim_recall_at_k": {
                str(cutoff): self.claim_recall_at_k[cutoff] for cutoff in self.cutoffs
            },
            "evidence_document_recall_at_k": {
                str(cutoff): self.evidence_document_recall_at_k[cutoff]
                for cutoff in self.cutoffs
            },
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a traceable report with the frozen ranked document IDs."""
        return {
            **self.summary_dict(),
            "predictions": [prediction.as_dict() for prediction in self.predictions],
        }


def _normalise_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    normalised = tuple(sorted(set(cutoffs)))
    if not normalised or any(
        isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0
        for cutoff in normalised
    ):
        raise RetrievalEvaluationError("Retrieval cutoffs must be positive integers.")
    return normalised


def retrieve_claims(
    index: BM25Index, claims: Sequence[Claim], *, top_k: int
) -> tuple[RetrievalPrediction, ...]:
    """Run retrieval using only safe runtime claims and the public corpus index."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalEvaluationError("Retrieval top_k must be a positive integer.")
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise RetrievalEvaluationError("Runtime claim IDs must be unique.")
    return tuple(
        RetrievalPrediction(claim_id=claim.claim_id, hits=index.search(claim.text, top_k))
        for claim in claims
    )


def evaluate_retrieval_predictions(
    predictions: Sequence[RetrievalPrediction],
    gold_evidence_documents: Mapping[int, frozenset[int]],
    *,
    cutoffs: Sequence[int],
) -> RetrievalEvaluationResult:
    """Evaluate already-frozen retrieval predictions against evaluator-only gold."""
    normalised_cutoffs = _normalise_cutoffs(cutoffs)
    prediction_by_claim: dict[int, RetrievalPrediction] = {}
    for prediction in predictions:
        if prediction.claim_id in prediction_by_claim:
            raise RetrievalEvaluationError(
                f"Duplicate prediction for claim ID {prediction.claim_id}."
            )
        prediction_by_claim[prediction.claim_id] = prediction

    missing_predictions = sorted(set(gold_evidence_documents) - set(prediction_by_claim))
    if missing_predictions:
        raise RetrievalEvaluationError(
            f"Missing retrieval prediction(s) for claim IDs: {missing_predictions}."
        )

    gold_bearing_claims = {
        claim_id: document_ids
        for claim_id, document_ids in gold_evidence_documents.items()
        if document_ids
    }
    if not gold_bearing_claims:
        raise RetrievalEvaluationError("No claims with gold evidence are available for Recall@k.")

    claim_hits = {cutoff: 0 for cutoff in normalised_cutoffs}
    evidence_hits = {cutoff: 0 for cutoff in normalised_cutoffs}
    reciprocal_ranks: list[float] = []
    total_gold_documents = sum(len(document_ids) for document_ids in gold_bearing_claims.values())

    for claim_id, gold_document_ids in gold_bearing_claims.items():
        retrieved_doc_ids = [hit.doc_id for hit in prediction_by_claim[claim_id].hits]
        first_relevant_rank = next(
            (
                rank
                for rank, doc_id in enumerate(retrieved_doc_ids, start=1)
                if doc_id in gold_document_ids
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant_rank is None else 1 / first_relevant_rank)
        for cutoff in normalised_cutoffs:
            retrieved_at_cutoff = set(retrieved_doc_ids[:cutoff])
            relevant_at_cutoff = retrieved_at_cutoff.intersection(gold_document_ids)
            claim_hits[cutoff] += int(bool(relevant_at_cutoff))
            evidence_hits[cutoff] += len(relevant_at_cutoff)

    denominator = len(gold_bearing_claims)
    return RetrievalEvaluationResult(
        claim_count=len(prediction_by_claim),
        gold_bearing_claim_count=denominator,
        claims_without_gold_evidence=len(gold_evidence_documents) - denominator,
        cutoffs=normalised_cutoffs,
        claim_recall_at_k={cutoff: claim_hits[cutoff] / denominator for cutoff in normalised_cutoffs},
        evidence_document_recall_at_k={
            cutoff: evidence_hits[cutoff] / total_gold_documents
            for cutoff in normalised_cutoffs
        },
        mean_reciprocal_rank=sum(reciprocal_ranks) / denominator,
        predictions=tuple(predictions),
    )


def write_retrieval_report(payload: Mapping[str, object], path: Path) -> None:
    """Write a stable machine-readable retrieval report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
