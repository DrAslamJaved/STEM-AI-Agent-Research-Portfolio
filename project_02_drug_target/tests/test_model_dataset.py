import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.representations import FEATURE_COLUMNS
from src.models.dataset import (
    ModelInputError,
    audit_model_inputs,
    load_train_test_data,
    write_model_input_audit,
)


LABEL_COLUMN = "interaction_kd_le_1000_nM"


def synthetic_feature_table() -> pd.DataFrame:
    """Return two labels in each partition with no valid cold-drug overlap."""
    rows = 8

    table = pd.DataFrame(
        {
            "observed_pair_index": list(range(rows)),
            "drug_id": [
                "drug_a",
                "drug_a",
                "drug_b",
                "drug_b",
                "drug_c",
                "drug_c",
                "drug_d",
                "drug_d",
            ],
            "target_id": [
                "target_1",
                "target_2",
                "target_1",
                "target_2",
                "target_1",
                "target_2",
                "target_1",
                "target_2",
            ],
            LABEL_COLUMN: [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    for feature_number, column in enumerate(FEATURE_COLUMNS):
        table[column] = np.arange(rows, dtype=float) + feature_number

    return table


def synthetic_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_policy": ["cold_drug"] * 8,
            "observed_pair_index": list(range(8)),
            "partition": [
                "train",
                "train",
                "train",
                "train",
                "test",
                "test",
                "test",
                "test",
            ],
        }
    )


def test_loader_returns_feature_only_matrices_and_metadata() -> None:
    dataset = load_train_test_data(
        synthetic_feature_table(),
        synthetic_assignments(),
        label_column=LABEL_COLUMN,
        policy="cold_drug",
    )

    assert tuple(dataset.X_train.columns) == FEATURE_COLUMNS
    assert tuple(dataset.X_test.columns) == FEATURE_COLUMNS

    assert LABEL_COLUMN not in dataset.X_train.columns
    assert "drug_id" not in dataset.X_train.columns
    assert "target_id" not in dataset.X_train.columns
    assert "observed_pair_index" not in dataset.X_train.columns

    assert len(dataset.X_train) == 4
    assert len(dataset.X_test) == 4

    assert dataset.y_train.tolist() == [0, 1, 0, 1]

    assert dataset.test_metadata["drug_id"].tolist() == [
        "drug_c",
        "drug_c",
        "drug_d",
        "drug_d",
    ]


def test_loader_rejects_assignment_coverage_gaps() -> None:
    assignments = synthetic_assignments().iloc[:-1].copy()

    with pytest.raises(
        ModelInputError,
        match="do not cover feature-table indices",
    ):
        load_train_test_data(
            synthetic_feature_table(),
            assignments,
            label_column=LABEL_COLUMN,
            policy="cold_drug",
        )


def test_loader_rejects_duplicate_assignments_within_policy() -> None:
    assignments = pd.concat(
        [synthetic_assignments(), synthetic_assignments().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ModelInputError,
        match="duplicate observed_pair_index",
    ):
        load_train_test_data(
            synthetic_feature_table(),
            assignments,
            label_column=LABEL_COLUMN,
            policy="cold_drug",
        )


def test_loader_rejects_invalid_partition_label() -> None:
    assignments = synthetic_assignments()
    assignments.loc[0, "partition"] = "validation"

    with pytest.raises(
        ModelInputError,
        match="only train and test",
    ):
        load_train_test_data(
            synthetic_feature_table(),
            assignments,
            label_column=LABEL_COLUMN,
            policy="cold_drug",
        )


def test_loader_rejects_cold_drug_overlap() -> None:
    assignments = synthetic_assignments()

    assignments.loc[1, "partition"] = "test"
    assignments.loc[4, "partition"] = "train"

    with pytest.raises(
        ModelInputError,
        match="overlapping drug",
    ):
        load_train_test_data(
            synthetic_feature_table(),
            assignments,
            label_column=LABEL_COLUMN,
            policy="cold_drug",
        )


def test_loader_rejects_nonfinite_feature_values() -> None:
    feature_table = synthetic_feature_table()
    feature_table.loc[0, FEATURE_COLUMNS[0]] = np.nan

    with pytest.raises(
        ModelInputError,
        match="missing values",
    ):
        load_train_test_data(
            feature_table,
            synthetic_assignments(),
            label_column=LABEL_COLUMN,
            policy="cold_drug",
        )


def test_audit_writer_is_reproducible(tmp_path: Path) -> None:
    dataset = load_train_test_data(
        synthetic_feature_table(),
        synthetic_assignments(),
        label_column=LABEL_COLUMN,
        policy="cold_drug",
    )

    audit = audit_model_inputs(dataset)

    path = write_model_input_audit(audit, tmp_path / "audit.json")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["policy"] == "cold_drug"
    assert payload["feature_column_count"] == len(FEATURE_COLUMNS)
    assert payload["drug_overlap_count"] == 0
    assert payload["train_positive_count"] == 2
    assert payload["test_positive_count"] == 2