"""Fixed hybrid retrieval and transparent candidate reranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidence_agent.data.schemas import Claim
from evidence_agent.evaluation.retrieval import RetrievalPrediction
from evidence_agent.retrieval.bm25 import BM25Index, RetrievalHit, tokenize
from evidence_agent.retrieval.scifact import CorpusDocument
from evidence_agent.retrieval.semantic import LsaSemanticIndex, SemanticHit


DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_RANK_CONSTANT = 60
RERANK_WEIGHTS = {
    "bm25_score": 0.15,
    "rrf_score": 0.45,
    "semantic_score": 0.35,
    "title_term_coverage": 0.05,
}


class HybridRetrievalError(ValueError):
    """Raised when hybrid components cannot produce a valid frozen ranking."""


@dataclass(frozen=True, slots=True)
class HybridHit:
    """A final document rank with its fixed fusion and reranking diagnostics."""

    doc_id: int
    score: float
    bm25_rank: int | None
    bm25_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    rrf_score: float
    title_term_coverage: float

    def as_dict(self) -> dict[str, object]:
        return {
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "doc_id": self.doc_id,
            "rerank_score": self.score,
            "rrf_score": self.rrf_score,
            "semantic_rank": self.semantic_rank,
            "semantic_score": self.semantic_score,
            "title_term_coverage": self.title_term_coverage,
        }


@dataclass(frozen=True, slots=True)
class HybridPrediction:
    """Frozen hybrid runtime output for one leakage-safe claim."""

    claim_id: int
    hits: tuple[HybridHit, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "retrieved_documents": [hit.as_dict() for hit in self.hits],
        }

    def as_evaluation_prediction(self) -> RetrievalPrediction:
        """Expose only document IDs and final scores to the generic evaluator."""
        return RetrievalPrediction(
            claim_id=self.claim_id,
            hits=tuple(RetrievalHit(doc_id=hit.doc_id, score=hit.score) for hit in self.hits),
        )


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HybridRetrievalError(f"{name} must be a positive integer.")


def _min_max_normalise(values: Mapping[int, float | None]) -> dict[int, float]:
    present = [value for value in values.values() if value is not None]
    if not present:
        return {doc_id: 0.0 for doc_id in values}
    lower, upper = min(present), max(present)
    if upper == lower:
        return {
            doc_id: float(value is not None)
            for doc_id, value in values.items()
        }
    return {
        doc_id: 0.0 if value is None else (value - lower) / (upper - lower)
        for doc_id, value in values.items()
    }


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """Fuse BM25 and LSA ranks, then rerank their public-corpus candidates."""

    bm25_index: BM25Index
    semantic_index: LsaSemanticIndex
    corpus: Mapping[int, CorpusDocument]
    candidate_k: int = DEFAULT_CANDIDATE_K
    rrf_rank_constant: int = DEFAULT_RRF_RANK_CONSTANT

    def __post_init__(self) -> None:
        _validate_positive_int(self.candidate_k, "candidate_k")
        _validate_positive_int(self.rrf_rank_constant, "rrf_rank_constant")
        corpus_doc_ids = set(self.corpus)
        bm25_doc_ids = set(self.bm25_index.document_lengths)
        semantic_doc_ids = {int(doc_id) for doc_id in self.semantic_index.doc_ids}
        if corpus_doc_ids != bm25_doc_ids or corpus_doc_ids != semantic_doc_ids:
            raise HybridRetrievalError(
                "Hybrid corpus and both retrieval indexes must contain identical document IDs."
            )
        if self.bm25_index.corpus_sha256 != self.semantic_index.corpus_sha256:
            raise HybridRetrievalError("Hybrid retrieval indexes have different corpus SHA-256 values.")

    def settings_dict(self) -> dict[str, object]:
        """Return fixed, pre-evaluation hybrid settings for the result manifest."""
        return {
            "candidate_k": self.candidate_k,
            "fusion": "reciprocal_rank_fusion",
            "rerank_weights": dict(RERANK_WEIGHTS),
            "rrf_rank_constant": self.rrf_rank_constant,
        }

    def search(self, claim: Claim, k: int) -> tuple[HybridHit, ...]:
        """Retrieve and rerank public documents for a safe ``Claim`` object only."""
        _validate_positive_int(k, "Retrieval k")
        bm25_hits = self.bm25_index.search(claim.text, self.candidate_k)
        semantic_hits = self.semantic_index.search(claim.text, self.candidate_k)
        bm25_by_doc = {hit.doc_id: hit for hit in bm25_hits}
        semantic_by_doc = {hit.doc_id: hit for hit in semantic_hits}
        candidate_doc_ids = sorted(set(bm25_by_doc).union(semantic_by_doc))
        if not candidate_doc_ids:
            return ()

        bm25_ranks = {hit.doc_id: rank for rank, hit in enumerate(bm25_hits, start=1)}
        semantic_ranks = {
            hit.doc_id: rank for rank, hit in enumerate(semantic_hits, start=1)
        }
        rrf_scores = {
            doc_id: sum(
                1 / (self.rrf_rank_constant + rank)
                for rank in (bm25_ranks.get(doc_id), semantic_ranks.get(doc_id))
                if rank is not None
            )
            for doc_id in candidate_doc_ids
        }
        bm25_scores = {
            doc_id: bm25_by_doc[doc_id].score if doc_id in bm25_by_doc else None
            for doc_id in candidate_doc_ids
        }
        semantic_scores = {
            doc_id: semantic_by_doc[doc_id].score if doc_id in semantic_by_doc else None
            for doc_id in candidate_doc_ids
        }
        normalised_bm25 = _min_max_normalise(bm25_scores)
        normalised_semantic = _min_max_normalise(semantic_scores)
        maximum_rrf = max(rrf_scores.values())
        query_terms = set(tokenize(claim.text))

        reranked: list[HybridHit] = []
        for doc_id in candidate_doc_ids:
            title_terms = set(tokenize(self.corpus[doc_id].title))
            title_coverage = (
                len(query_terms.intersection(title_terms)) / len(query_terms)
                if query_terms
                else 0.0
            )
            rrf_normalised = rrf_scores[doc_id] / maximum_rrf
            rerank_score = (
                RERANK_WEIGHTS["rrf_score"] * rrf_normalised
                + RERANK_WEIGHTS["semantic_score"] * normalised_semantic[doc_id]
                + RERANK_WEIGHTS["bm25_score"] * normalised_bm25[doc_id]
                + RERANK_WEIGHTS["title_term_coverage"] * title_coverage
            )
            bm25_hit = bm25_by_doc.get(doc_id)
            semantic_hit = semantic_by_doc.get(doc_id)
            reranked.append(
                HybridHit(
                    doc_id=doc_id,
                    score=rerank_score,
                    bm25_rank=bm25_ranks.get(doc_id),
                    bm25_score=None if bm25_hit is None else bm25_hit.score,
                    semantic_rank=semantic_ranks.get(doc_id),
                    semantic_score=None if semantic_hit is None else semantic_hit.score,
                    rrf_score=rrf_scores[doc_id],
                    title_term_coverage=title_coverage,
                )
            )
        reranked.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return tuple(reranked[:k])


def retrieve_hybrid_claims(
    retriever: HybridRetriever, claims: Sequence[Claim], *, top_k: int
) -> tuple[HybridPrediction, ...]:
    """Freeze hybrid rankings using only runtime-safe claim objects."""
    _validate_positive_int(top_k, "Retrieval top_k")
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise HybridRetrievalError("Runtime claim IDs must be unique.")
    return tuple(
        HybridPrediction(claim_id=claim.claim_id, hits=retriever.search(claim, top_k))
        for claim in claims
    )


def as_evaluation_predictions(
    predictions: Sequence[HybridPrediction],
) -> tuple[RetrievalPrediction, ...]:
    """Convert frozen hybrid outputs into the generic Recall@k evaluator shape."""
    return tuple(prediction.as_evaluation_prediction() for prediction in predictions)
