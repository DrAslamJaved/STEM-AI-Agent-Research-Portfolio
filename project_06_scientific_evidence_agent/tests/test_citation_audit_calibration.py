"""Tests for train-only fold filtering and citation-audit selection."""

from __future__ import annotations

from pathlib import Path

from evidence_agent.audit.calibration import (
    PolicyScore,
    calibrate_citation_audit,
    derive_train_only_fold_partitions,
    load_selected_policy,
    select_policy,
    write_calibration_report,
)
from evidence_agent.audit.policy import CitationAuditPolicy
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.retrieval.bm25 import build_bm25_index
from evidence_agent.retrieval.scifact import load_scifact_corpus
from tests.helpers import write_citation_audit_scifact_dataset


def _summary(*, unsupported: float, coverage: float, faithfulness: float = 0.0) -> dict:
    return {
        "assertive_decision_count": int(10 * coverage),
        "citation_correctness": {"f1": 0.1},
        "claim_classification": {"macro_f1": 0.2},
        "coverage": coverage,
        "evidence_sentence": {"f1": 0.1},
        "faithfulness": faithfulness,
        "unsupported_assertion_rate": unsupported,
    }


def test_supplied_fold_assignments_are_filtered_to_main_training_claims(tmp_path: Path) -> None:
    dataset = write_citation_audit_scifact_dataset(tmp_path / "scifact")

    partitions = derive_train_only_fold_partitions(
        dataset / "claims_train.jsonl",
        dataset / "claims_dev.jsonl",
        dataset / "cross_validation",
    )

    assert len(partitions) == 5
    assert {claim_id for partition in partitions for claim_id in partition.validation_claim_ids} == set(
        range(1, 11)
    )
    assert all(100 not in partition.validation_claim_ids for partition in partitions)
    assert all(partition.ordinary_development_claim_count_excluded == 1 for partition in partitions)
    assert all(len(partition.training_claim_ids) == 8 for partition in partitions)


def test_selection_enforces_coverage_before_minimising_unsupported_rate() -> None:
    abstaining = PolicyScore(
        CitationAuditPolicy(0.95, 0.95, 1),
        _summary(unsupported=0.0, coverage=0.0),
    )
    useful = PolicyScore(
        CitationAuditPolicy(0.75, 0.75, 1),
        _summary(unsupported=0.20, coverage=0.20, faithfulness=0.5),
    )
    selected = select_policy((abstaining, useful), minimum_coverage=0.20)

    assert selected == useful.policy


def test_calibration_writes_fold_traces_before_loading_train_fold_metrics(tmp_path: Path) -> None:
    dataset = write_citation_audit_scifact_dataset(tmp_path / "scifact")
    corpus = load_scifact_corpus(dataset / "corpus.jsonl")
    index = build_bm25_index(
        {doc_id: document.searchable_text for doc_id, document in corpus.items()},
        corpus_sha256=sha256_file(dataset / "corpus.jsonl"),
    )
    calibration, policy_grid = calibrate_citation_audit(
        corpus_path=dataset / "corpus.jsonl",
        main_training_claims_path=dataset / "claims_train.jsonl",
        ordinary_development_claims_path=dataset / "claims_dev.jsonl",
        cross_validation_dir=dataset / "cross_validation",
        index=index,
        artifact_dir=tmp_path / "artifacts",
        assertion_thresholds=[0.0],
        sentence_thresholds=[0.0],
        max_sentences_per_citation=[1],
        minimum_coverage=0.0,
        max_features=100,
        retrieval_k=2,
    )

    assert len(policy_grid) == 1
    assert calibration.selected_policy == policy_grid[0]
    assert all(artifact.model_path.is_file() for artifact in calibration.fold_artifacts)
    assert all(artifact.trace_path.is_file() for artifact in calibration.fold_artifacts)
    assert all(len(artifact.traces) == 2 for artifact in calibration.fold_artifacts)

    payload = calibration.as_dict(
        corpus_sha256=sha256_file(dataset / "corpus.jsonl"),
        main_training_claims_path=dataset / "claims_train.jsonl",
        ordinary_development_claims_path=dataset / "claims_dev.jsonl",
        cross_validation_dir=dataset / "cross_validation",
        index=index,
        policy_grid=policy_grid,
        retrieval_k=2,
        random_seed=20260904,
        max_features=100,
    )
    report_path = tmp_path / "results" / "calibration.json"
    write_calibration_report(payload, report_path)

    assert payload["cross_validation"]["development_claims_excluded_from_selection"]
    assert load_selected_policy(report_path) == policy_grid[0]
