"""Tests for the runtime-only BM25 -> sentence -> stance verification path."""

from __future__ import annotations

from evidence_agent.data.acquisition import sha256_file
from evidence_agent.retrieval.bm25 import build_bm25_index
from evidence_agent.retrieval.scifact import load_runtime_claims, load_scifact_corpus
from evidence_agent.verification.agent import run_verification_agent
from evidence_agent.verification.models import fit_verifier_bundle
from evidence_agent.verification.scifact import load_verification_training_data
from tests.helpers import write_verification_scifact_dataset


def _contains_key(value, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_runtime_agent_emits_traceable_decisions_without_gold_fields(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")
    corpus = load_scifact_corpus(dataset / "corpus.jsonl")
    training = load_verification_training_data(
        dataset / "claims_train.jsonl", dataset / "corpus.jsonl"
    )
    bundle = fit_verifier_bundle(
        training.stance_examples,
        training.sentence_examples,
        training_claims_sha256=sha256_file(dataset / "claims_train.jsonl"),
        corpus_sha256=sha256_file(dataset / "corpus.jsonl"),
        max_features=100,
    )
    index = build_bm25_index(
        {doc_id: document.searchable_text for doc_id, document in corpus.items()},
        corpus_sha256=sha256_file(dataset / "corpus.jsonl"),
    )

    traces = run_verification_agent(
        bundle,
        index,
        corpus,
        load_runtime_claims(dataset / "claims_dev.jsonl"),
        retrieval_k=2,
        assertion_threshold=0.0,
        sentence_threshold=0.0,
    )

    assert len(traces) == 3
    assert all(trace.decision.claim_id in {4, 5, 6} for trace in traces)
    serialized = traces[0].as_dict()
    assert not _contains_key(serialized, "evidence")
    assert not _contains_key(serialized, "cited_doc_ids")
    assert serialized["candidates"]

    decision = traces[0].decision_dict()
    assert set(decision) == {"claim_id", "verdict", "confidence", "citations"}
    assert not _contains_key(decision, "evidence")
    assert not _contains_key(decision, "cited_doc_ids")
