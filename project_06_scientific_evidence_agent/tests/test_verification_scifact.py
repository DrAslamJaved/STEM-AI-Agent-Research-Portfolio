"""Tests for train/evaluation-only SciFact verification adapters."""

from __future__ import annotations

import json

from evidence_agent.data.schemas import Verdict
from evidence_agent.verification.scifact import (
    load_gold_claim_annotations,
    load_stance_benchmark,
    load_stance_benchmark_inputs,
    load_stance_benchmark_labels,
    load_verification_training_data,
)
from tests.helpers import write_verification_scifact_dataset


def test_training_loader_builds_three_way_and_sentence_examples(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")

    training = load_verification_training_data(
        dataset / "claims_train.jsonl", dataset / "corpus.jsonl"
    )

    assert training.claim_count == 3
    assert len(training.stance_examples) == 3
    assert len(training.sentence_examples) == 6
    assert training.summary_dict()["stance_label_counts"] == {
        "CONTRADICT": 1,
        "NO_EVIDENCE": 1,
        "SUPPORT": 1,
    }
    assert sum(example.is_evidence for example in training.sentence_examples) == 2


def test_benchmark_inputs_do_not_duplicate_cited_document_ids(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")

    inputs = load_stance_benchmark_inputs(dataset / "claims_dev.jsonl", dataset / "corpus.jsonl")
    labels = load_stance_benchmark_labels(dataset / "claims_dev.jsonl", dataset / "corpus.jsonl")
    combined_inputs, combined_labels = load_stance_benchmark(
        dataset / "claims_dev.jsonl", dataset / "corpus.jsonl"
    )

    assert len(inputs) == 3
    assert labels == (Verdict.SUPPORT, Verdict.CONTRADICT, Verdict.NO_EVIDENCE)
    assert combined_inputs == inputs
    assert combined_labels == labels


def test_gold_claim_annotations_preserve_complete_rationales(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")

    annotations = load_gold_claim_annotations(dataset / "claims_dev.jsonl", dataset / "corpus.jsonl")

    assert annotations[4].verdict is Verdict.SUPPORT
    assert annotations[4].citations[0].sentence_ids == (0,)
    assert annotations[5].verdict is Verdict.CONTRADICT
    assert annotations[6].verdict is Verdict.NO_EVIDENCE
    assert annotations[6].citations == ()


def test_claim_id_filter_skips_excluded_gold_records_before_they_are_parsed(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")
    training_path = dataset / "claims_train.jsonl"
    records = [json.loads(line) for line in training_path.read_text(encoding="utf-8").splitlines()]
    records[2]["evidence"] = "not a valid evidence object"
    training_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    training = load_verification_training_data(
        training_path,
        dataset / "corpus.jsonl",
        claim_ids={1, 2},
    )
    annotations = load_gold_claim_annotations(
        dataset / "claims_dev.jsonl",
        dataset / "corpus.jsonl",
        claim_ids={4},
    )

    assert training.claim_count == 2
    assert set(annotations) == {4}
