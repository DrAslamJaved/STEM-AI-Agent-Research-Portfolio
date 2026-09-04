"""Structural tests for SciFact data validation and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence_agent.data.scifact import (
    DatasetValidationError,
    validate_scifact_dataset,
    write_validation_report,
)
from tests.helpers import write_minimal_scifact_dataset


def test_validate_scifact_dataset_returns_expected_summary(tmp_path: Path) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "release", include_cross_validation=True)

    summary = validate_scifact_dataset(dataset)

    assert summary.corpus_documents == 2
    assert summary.corpus_sentences == 3
    assert summary.claims_by_split == {"train": 1, "dev": 1, "test": 1}
    assert summary.evidence_labels == {"CONTRADICT": 1, "SUPPORT": 1}
    assert summary.cross_validation_folds == 5


def test_validate_scifact_dataset_rejects_out_of_range_rationale_sentence(
    tmp_path: Path,
) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "release")
    claims_path = dataset / "claims_train.jsonl"
    record = json.loads(claims_path.read_text(encoding="utf-8"))
    record["evidence"]["10"][0]["sentences"] = [9]
    claims_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="outside document 10"):
        validate_scifact_dataset(dataset, require_cross_validation=False)


def test_validation_report_is_json_and_sorted(tmp_path: Path) -> None:
    dataset = write_minimal_scifact_dataset(tmp_path / "release")
    summary = validate_scifact_dataset(dataset, require_cross_validation=False)
    report_path = tmp_path / "validation" / "report.json"

    write_validation_report(summary, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["claims_by_split"] == {"dev": 1, "test": 1, "train": 1}
