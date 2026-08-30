import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.data.validate_davis import (
    DavisValidationError,
    validate_davis_dataset,
    write_validation_report,
)


def write_synthetic_davis(
    data_dir: Path,
    *,
    matrix: np.ndarray | None = None,
    ligands: dict[str, str] | None = None,
    proteins: dict[str, str] | None = None,
    train_indices: list[list[int]] | None = None,
    test_indices: list[int] | None = None,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "folds").mkdir(exist_ok=True)

    ligands = ligands or {"drug_a": "CCO", "drug_b": "CCC"}
    proteins = proteins or {
        "target_a": "MKT",
        "target_b": "MLA",
        "target_c": "MGH",
    }
    matrix = (
        matrix
        if matrix is not None
        else np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    )
    train_indices = train_indices or [[1, 2], [3, 4]]
    test_indices = test_indices or [0, 5]

    (data_dir / "ligands_can.txt").write_text(json.dumps(ligands), encoding="utf-8")
    (data_dir / "proteins.txt").write_text(json.dumps(proteins), encoding="utf-8")
    with (data_dir / "Y").open("wb") as handle:
        pickle.dump(matrix, handle)

    (data_dir / "folds" / "train_fold_setting1.txt").write_text(
        json.dumps(train_indices), encoding="utf-8"
    )
    (data_dir / "folds" / "test_fold_setting1.txt").write_text(
        json.dumps(test_indices), encoding="utf-8"
    )
    return data_dir


def test_valid_davis_dataset_returns_expected_summary(tmp_path: Path) -> None:
    report = validate_davis_dataset(write_synthetic_davis(tmp_path / "davis"))

    assert report.drug_count == 2
    assert report.target_count == 3
    assert report.affinity_shape == (2, 3)
    assert report.total_pair_count == 6
    assert report.affinity_missing_values == 0
    assert report.fold_overlap_count == 0
    assert report.fold_coverage_count == 6


def test_validator_writes_machine_readable_report(tmp_path: Path) -> None:
    report = validate_davis_dataset(write_synthetic_davis(tmp_path / "davis"))
    output_path = write_validation_report(report, tmp_path / "report.json")

    saved_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_report["drug_count"] == 2
    assert saved_report["affinity_shape"] == [2, 3]


def test_validator_rejects_matrix_dimension_mismatch(tmp_path: Path) -> None:
    data_dir = write_synthetic_davis(
        tmp_path / "davis",
        matrix=np.array([[1.0, 2.0, 3.0]]),
    )

    with pytest.raises(DavisValidationError, match="shape does not match"):
        validate_davis_dataset(data_dir)


def test_validator_rejects_out_of_range_fold_index(tmp_path: Path) -> None:
    data_dir = write_synthetic_davis(tmp_path / "davis", test_indices=[6])

    with pytest.raises(DavisValidationError, match="out-of-range"):
        validate_davis_dataset(data_dir)


def test_validator_rejects_train_test_overlap(tmp_path: Path) -> None:
    data_dir = write_synthetic_davis(tmp_path / "davis", test_indices=[0, 1])

    with pytest.raises(DavisValidationError, match="overlap"):
        validate_davis_dataset(data_dir)


def test_validator_reports_missing_affinity_values(tmp_path: Path) -> None:
    data_dir = write_synthetic_davis(
        tmp_path / "davis",
        matrix=np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
    )

    report = validate_davis_dataset(data_dir)

    assert report.affinity_missing_values == 1
    assert report.affinity_non_missing_values == 5