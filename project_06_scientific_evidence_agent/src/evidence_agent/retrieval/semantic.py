"""Deterministic latent-semantic retrieval over the public SciFact corpus."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


SEMANTIC_INDEX_FORMAT = "evidence_agent_lsa_v1"


class SemanticIndexError(ValueError):
    """Raised when a latent-semantic index or its inputs are invalid."""


@dataclass(frozen=True, slots=True)
class SemanticHit:
    """One document and cosine similarity returned by latent-semantic search."""

    doc_id: int
    score: float

    def as_dict(self) -> dict[str, object]:
        return {"doc_id": self.doc_id, "score": self.score}


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SemanticIndexError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class LsaSemanticIndex:
    """A persisted TF-IDF + truncated-SVD semantic retrieval index."""

    doc_ids: np.ndarray
    document_embeddings: np.ndarray
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    corpus_sha256: str
    random_seed: int
    min_document_frequency: int

    def __post_init__(self) -> None:
        if self.doc_ids.ndim != 1 or self.document_embeddings.ndim != 2:
            raise SemanticIndexError("Semantic index arrays have invalid dimensions.")
        if len(self.doc_ids) == 0 or len(self.doc_ids) != len(self.document_embeddings):
            raise SemanticIndexError("Semantic index documents and embeddings are inconsistent.")
        if self.document_embeddings.shape[1] == 0:
            raise SemanticIndexError("Semantic index must contain latent dimensions.")
        if not isinstance(self.corpus_sha256, str) or len(self.corpus_sha256) != 64:
            raise SemanticIndexError("Semantic index must retain a SHA-256 corpus fingerprint.")
        _validate_positive_int(self.random_seed, "random_seed")
        _validate_positive_int(self.min_document_frequency, "min_document_frequency")

    @property
    def document_count(self) -> int:
        """Return the number of corpus documents represented in the index."""
        return int(len(self.doc_ids))

    @property
    def vocabulary_size(self) -> int:
        """Return the fitted TF-IDF vocabulary size."""
        return len(self.vectorizer.vocabulary_)

    @property
    def latent_dimensions(self) -> int:
        """Return the number of retained latent-semantic dimensions."""
        return int(self.document_embeddings.shape[1])

    def summary_dict(self) -> dict[str, object]:
        """Return reproducibility metadata without serializing model internals."""
        return {
            "algorithm": "tfidf_lsa",
            "corpus_sha256": self.corpus_sha256,
            "document_count": self.document_count,
            "format": SEMANTIC_INDEX_FORMAT,
            "latent_dimensions": self.latent_dimensions,
            "min_document_frequency": self.min_document_frequency,
            "random_seed": self.random_seed,
            "scikit_learn_version": sklearn.__version__,
            "vocabulary_size": self.vocabulary_size,
        }

    def search(self, query: str, k: int) -> tuple[SemanticHit, ...]:
        """Return the highest cosine-similarity document embeddings for *query*."""
        _validate_positive_int(k, "Retrieval k")
        if not isinstance(query, str) or not query.strip():
            raise SemanticIndexError("Semantic retrieval query must be non-empty text.")

        query_tfidf = self.vectorizer.transform([query])
        query_embedding = self.svd.transform(query_tfidf)
        query_norm = float(np.linalg.norm(query_embedding))
        if math.isclose(query_norm, 0.0):
            return ()
        query_embedding /= query_norm
        scores = self.document_embeddings @ query_embedding.ravel()
        ranking = np.lexsort((self.doc_ids, -scores))[:k]
        return tuple(
            SemanticHit(doc_id=int(self.doc_ids[position]), score=float(scores[position]))
            for position in ranking
        )


def build_lsa_index(
    documents: Mapping[int, str],
    *,
    corpus_sha256: str,
    n_components: int = 128,
    random_seed: int = 20260904,
    min_document_frequency: int = 2,
) -> LsaSemanticIndex:
    """Fit a deterministic corpus-only latent-semantic index.

    This unsupervised fit consumes public document content only.  SciFact claims,
    evidence labels, and citation annotations are not accepted by this function.
    """
    _validate_positive_int(n_components, "n_components")
    _validate_positive_int(random_seed, "random_seed")
    _validate_positive_int(min_document_frequency, "min_document_frequency")
    if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
        raise SemanticIndexError("corpus_sha256 must be a 64-character SHA-256 digest.")
    if len(documents) < 2:
        raise SemanticIndexError("LSA requires at least two public corpus documents.")

    doc_ids = np.asarray(sorted(documents), dtype=np.int64)
    texts: list[str] = []
    for doc_id in doc_ids:
        text = documents[int(doc_id)]
        if not isinstance(text, str) or not text.strip():
            raise SemanticIndexError(f"Semantic document {doc_id} must contain text.")
        texts.append(text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=min_document_frequency,
        ngram_range=(1, 2),
        norm="l2",
        strip_accents="unicode",
        sublinear_tf=True,
    )
    try:
        document_tfidf = vectorizer.fit_transform(texts)
    except ValueError as error:
        raise SemanticIndexError(f"Unable to fit the TF-IDF vocabulary: {error}") from error

    maximum_components = min(document_tfidf.shape) - 1
    if n_components > maximum_components:
        raise SemanticIndexError(
            "n_components must be less than both the public document count and "
            f"the TF-IDF vocabulary size; maximum is {maximum_components}."
        )
    svd = TruncatedSVD(
        algorithm="randomized",
        n_components=n_components,
        n_iter=7,
        random_state=random_seed,
    )
    embeddings = normalize(svd.fit_transform(document_tfidf), norm="l2", copy=False)
    return LsaSemanticIndex(
        doc_ids=doc_ids,
        document_embeddings=np.asarray(embeddings, dtype=np.float64),
        vectorizer=vectorizer,
        svd=svd,
        corpus_sha256=corpus_sha256,
        random_seed=random_seed,
        min_document_frequency=min_document_frequency,
    )


def write_lsa_index(index: LsaSemanticIndex, path: Path) -> None:
    """Persist a locally generated semantic index under an ignored artifact path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path, compress=3)


def load_lsa_index(path: Path) -> LsaSemanticIndex:
    """Load a trusted local LSA artifact written by :func:`write_lsa_index`."""
    path = Path(path)
    try:
        index = joblib.load(path)
    except (OSError, ValueError, EOFError) as error:
        raise SemanticIndexError(f"Unable to read semantic index {path}: {error}") from error
    if not isinstance(index, LsaSemanticIndex):
        raise SemanticIndexError("Semantic index artifact has an unexpected type.")
    return index
