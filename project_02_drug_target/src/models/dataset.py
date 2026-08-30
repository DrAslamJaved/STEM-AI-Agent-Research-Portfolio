"""Load leakage-audited Davis model inputs from frozen local artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.representations import FEATURE_COLUMNS


POLICIES = ("random_pair", "cold_drug", "cold_target")
PARTITIONS = ("train", "test")
METADATA_COLUMNS = ("observed_pair_index", "drug_id", "target_id")


class ModelInputError(ValueError):
    """Raised when model inputs do not satisfy leakage-safety invariants."""


@dataclass(frozen=True)
class ModelDataset:
    """Feature-only train/test matrices plus separate audit metadata."""

    policy: str
    label_column: str
    feature_columns: tuple[str, ...]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame


@dataclass(frozen=True)
class ModelInputAudit:
    """Compact evidence that one model input partition is correctly formed."""

    policy: str
    label_column: str
    feature_column_count: int
    feature_columns: tuple[str, ...]
    total_pair_count: int
    train_pair_count: int
    test_pair_count: int
    train_positive_count: int
    train_negative_count: int
    train_positive_rate: float
    test_positive_count: int
    test_negative_count: int
    test_positive_rate: float
    train_drug_count: int
    test_drug_count: int
    drug_overlap_count: int
    train_target_count: int
    test_target_count: int
    target_overlap_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record."""
        return asdict(self)


def _normalise_identifier_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    table_label: str,
) -> pd.DataFrame:
    """Return a copy with required non-empty identifiers normalized to strings."""
    missing_columns = set(columns).difference(table.columns)
    if missing_columns:
        raise ModelInputError(
            f"{table_label} is missing columns: {sorted(missing_columns)}"
        )

    normalized = table.copy()

    for column in columns:
        if normalized[column].isna().any():
            raise ModelInputError(
                f"{table_label} contains missing values in {column}."
            )

        normalized[column] = normalized[column].astype(str).str.strip()

        if normalized[column].eq("").any():
            raise ModelInputError(
                f"{table_label} contains empty values in {column}."
            )

    return normalized


def _normalise_observed_pair_index(
    table: pd.DataFrame,
    table_label: str,
    *,
    require_unique: bool,
) -> pd.DataFrame:
    """Normalize observed-pair keys to finite integer values."""
    if "observed_pair_index" not in table.columns:
        raise ModelInputError(
            f"{table_label} is missing observed_pair_index."
        )

    normalized = table.copy()
    indices = pd.to_numeric(
        normalized["observed_pair_index"],
        errors="coerce",
    )

    if indices.isna().any():
        raise ModelInputError(
            f"{table_label} contains a non-numeric observed_pair_index."
        )

    values = indices.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ModelInputError(
            f"{table_label} observed_pair_index values must be finite."
        )

    if not np.equal(values, np.floor(values)).all():
        raise ModelInputError(
            f"{table_label} observed_pair_index values must be integers."
        )

    normalized["observed_pair_index"] = indices.astype(np.int64)

    if require_unique and normalized["observed_pair_index"].duplicated().any():
        raise ModelInputError(
            f"{table_label} contains duplicate observed_pair_index values."
        )

    return normalized


def _validated_binary_labels(
    table: pd.DataFrame,
    label_column: str,
) -> pd.Series:
    """Return a binary label series or raise a validation error."""
    if label_column not in table.columns:
        raise ModelInputError(
            f"Feature table is missing label column: {label_column}"
        )

    labels = pd.to_numeric(table[label_column], errors="coerce")

    if labels.isna().any():
        raise ModelInputError(
            f"{label_column} contains missing or non-numeric labels."
        )

    invalid_values = set(labels.unique()).difference({0, 1})

    if invalid_values:
        raise ModelInputError(
            f"{label_column} must contain only binary values 0 and 1."
        )

    if labels.nunique() != 2:
        raise ModelInputError(
            f"{label_column} must contain both positive and negative examples."
        )

    return labels.astype("int8")


def _validated_feature_values(table: pd.DataFrame) -> pd.DataFrame:
    """Return fixed feature values after numeric and finiteness checks."""
    missing_columns = set(FEATURE_COLUMNS).difference(table.columns)

    if missing_columns:
        raise ModelInputError(
            "Feature table is missing model feature columns: "
            f"{sorted(missing_columns)}"
        )

    try:
        feature_values = table.loc[:, list(FEATURE_COLUMNS)].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ModelInputError(
            "Model feature columns must be numeric."
        ) from error

    if feature_values.isna().any().any():
        raise ModelInputError(
            "Model feature columns contain missing values."
        )

    values = feature_values.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ModelInputError(
            "Model feature columns contain non-finite values."
        )

    return feature_values.astype(float)


def _prepare_feature_table(
    feature_table: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Validate a feature table while retaining labels only outside X."""
    required_columns = set(METADATA_COLUMNS).union(
        {label_column},
        set(FEATURE_COLUMNS),
    )

    missing_columns = required_columns.difference(feature_table.columns)

    if missing_columns:
        raise ModelInputError(
            f"Feature table is missing columns: {sorted(missing_columns)}"
        )

    if feature_table.empty:
        raise ModelInputError("Feature table is empty.")

    prepared = _normalise_identifier_columns(
        feature_table,
        ("drug_id", "target_id"),
        "Feature table",
    )

    prepared = _normalise_observed_pair_index(
        prepared,
        "Feature table",
        require_unique=True,
    )

    if prepared.duplicated(["drug_id", "target_id"]).any():
        raise ModelInputError(
            "Feature table contains duplicate drug-target pairs."
        )

    prepared[label_column] = _validated_binary_labels(
        prepared,
        label_column,
    )

    prepared.loc[:, list(FEATURE_COLUMNS)] = _validated_feature_values(
        prepared
    )

    return prepared


def _prepare_split_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Validate saved row-level assignments before policy selection."""
    required_columns = {
        "split_policy",
        "observed_pair_index",
        "partition",
    }

    missing_columns = required_columns.difference(assignments.columns)

    if missing_columns:
        raise ModelInputError(
            f"Split assignments are missing columns: {sorted(missing_columns)}"
        )

    if assignments.empty:
        raise ModelInputError("Split assignments are empty.")

    prepared = _normalise_identifier_columns(
        assignments,
        ("split_policy", "partition"),
        "Split assignments",
    )

    return _normalise_observed_pair_index(
        prepared,
        "Split assignments",
        require_unique=False,
    )


def _validate_selected_assignments(
    assignments: pd.DataFrame,
    feature_table: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    """Return one complete policy assignment table aligned to feature rows."""
    if policy not in POLICIES:
        raise ModelInputError(
            f"Unknown split policy: {policy}. Choose one of {list(POLICIES)}."
        )

    selected = assignments.loc[
        assignments["split_policy"].eq(policy),
        ["observed_pair_index", "partition"],
    ].copy()

    if selected.empty:
        raise ModelInputError(
            f"No split assignments found for policy: {policy}"
        )

    if selected["observed_pair_index"].duplicated().any():
        raise ModelInputError(
            f"{policy} assignments contain duplicate observed_pair_index values."
        )

    invalid_partitions = sorted(
        set(selected["partition"]).difference(PARTITIONS)
    )

    if invalid_partitions:
        raise ModelInputError(
            f"{policy} assignments must use only train and test partitions."
        )

    if set(selected["partition"]) != set(PARTITIONS):
        raise ModelInputError(
            f"{policy} assignments must contain both train and test partitions."
        )

    feature_indices = set(feature_table["observed_pair_index"])
    assignment_indices = set(selected["observed_pair_index"])

    missing_count = len(feature_indices.difference(assignment_indices))
    extra_count = len(assignment_indices.difference(feature_indices))

    if missing_count or extra_count:
        raise ModelInputError(
            f"{policy} assignments do not cover feature-table indices "
            f"(missing={missing_count}, extra={extra_count})."
        )

    return selected


def _partition_frames(
    joined: pd.DataFrame,
    label_column: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Extract feature-only matrices, labels, and separate audit metadata."""
    train_frame = joined.loc[joined["partition"].eq("train")].copy()
    test_frame = joined.loc[joined["partition"].eq("test")].copy()

    if train_frame.empty or test_frame.empty:
        raise ModelInputError(
            "Train and test partitions must both be non-empty."
        )

    X_train = train_frame.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)
    X_test = test_frame.loc[:, list(FEATURE_COLUMNS)].reset_index(drop=True)

    y_train = train_frame[label_column].astype("int8").reset_index(drop=True)
    y_test = test_frame[label_column].astype("int8").reset_index(drop=True)

    train_metadata = train_frame.loc[
        :,
        list(METADATA_COLUMNS),
    ].reset_index(drop=True)

    test_metadata = test_frame.loc[
        :,
        list(METADATA_COLUMNS),
    ].reset_index(drop=True)

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ModelInputError(
            "Train and test partitions must each contain positive and "
            "negative examples."
        )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        train_metadata,
        test_metadata,
    )


def _validate_policy_entity_separation(
    train_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    policy: str,
) -> None:
    """Enforce entity-disjoint rules for cold-start policies."""
    drug_overlap = set(train_metadata["drug_id"]).intersection(
        test_metadata["drug_id"]
    )

    target_overlap = set(train_metadata["target_id"]).intersection(
        test_metadata["target_id"]
    )

    if policy == "cold_drug" and drug_overlap:
        raise ModelInputError(
            "cold_drug partition contains overlapping drug identifiers."
        )

    if policy == "cold_target" and target_overlap:
        raise ModelInputError(
            "cold_target partition contains overlapping target identifiers."
        )


def load_train_test_data(
    feature_table: pd.DataFrame,
    split_assignments: pd.DataFrame,
    *,
    label_column: str,
    policy: str = "cold_drug",
) -> ModelDataset:
    """Build one leakage-checked, feature-only train/test dataset.

    The saved assignment table controls the partition. This function never
    calls a splitter and never fits a transformer, sampler, or model.
    """
    prepared_features = _prepare_feature_table(
        feature_table,
        label_column,
    )

    prepared_assignments = _prepare_split_assignments(split_assignments)

    selected_assignments = _validate_selected_assignments(
        prepared_assignments,
        prepared_features,
        policy,
    )

    joined = prepared_features.merge(
        selected_assignments,
        on="observed_pair_index",
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(joined) != len(prepared_features):
        raise ModelInputError(
            "Joining split assignments changed the feature-table row count."
        )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        train_metadata,
        test_metadata,
    ) = _partition_frames(joined, label_column)

    _validate_policy_entity_separation(
        train_metadata,
        test_metadata,
        policy,
    )

    return ModelDataset(
        policy=policy,
        label_column=label_column,
        feature_columns=tuple(FEATURE_COLUMNS),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_metadata=train_metadata,
        test_metadata=test_metadata,
    )


def audit_model_inputs(dataset: ModelDataset) -> ModelInputAudit:
    """Summarize class balance and entity overlap for prepared model inputs."""
    train_positive_count = int(dataset.y_train.sum())
    test_positive_count = int(dataset.y_test.sum())

    train_drugs = set(dataset.train_metadata["drug_id"])
    test_drugs = set(dataset.test_metadata["drug_id"])

    train_targets = set(dataset.train_metadata["target_id"])
    test_targets = set(dataset.test_metadata["target_id"])

    return ModelInputAudit(
        policy=dataset.policy,
        label_column=dataset.label_column,
        feature_column_count=len(dataset.feature_columns),
        feature_columns=dataset.feature_columns,
        total_pair_count=int(len(dataset.X_train) + len(dataset.X_test)),
        train_pair_count=int(len(dataset.X_train)),
        test_pair_count=int(len(dataset.X_test)),
        train_positive_count=train_positive_count,
        train_negative_count=int(len(dataset.y_train) - train_positive_count),
        train_positive_rate=float(
            train_positive_count / len(dataset.y_train)
        ),
        test_positive_count=test_positive_count,
        test_negative_count=int(len(dataset.y_test) - test_positive_count),
        test_positive_rate=float(
            test_positive_count / len(dataset.y_test)
        ),
        train_drug_count=int(len(train_drugs)),
        test_drug_count=int(len(test_drugs)),
        drug_overlap_count=int(len(train_drugs.intersection(test_drugs))),
        train_target_count=int(len(train_targets)),
        test_target_count=int(len(test_targets)),
        target_overlap_count=int(
            len(train_targets.intersection(test_targets))
        ),
    )


def write_model_input_audit(
    audit: ModelInputAudit,
    output_path: str | Path,
) -> Path:
    """Write compact version-controlled model-input evidence."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return destination


def main(argv: list[str] | None = None) -> int:
    """Run the audit from local ignored data artifacts."""
    parser = argparse.ArgumentParser(
        description="Load and audit fixed Davis model-input partitions."
    )

    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("data/processed/davis_pair_features.csv"),
        help="Local feature-table CSV from src.features.representations.",
    )

    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=Path("data/interim/davis_split_assignments.csv"),
        help="Local frozen split-assignment CSV from src.data.splits.",
    )

    parser.add_argument(
        "--label-column",
        default="interaction_kd_le_1000_nM",
        help="Pre-specified binary label column exposed as y only.",
    )

    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="cold_drug",
        help="Saved split policy to load without rebuilding it.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_model_input_audit.json"),
        help="Version-controlled JSON audit destination.",
    )

    args = parser.parse_args(argv)

    try:
        feature_table = pd.read_csv(
            args.feature_table,
            dtype={"drug_id": str, "target_id": str},
        )

        split_assignments = pd.read_csv(
            args.split_assignments,
            dtype={"split_policy": str, "partition": str},
        )

        dataset = load_train_test_data(
            feature_table,
            split_assignments,
            label_column=args.label_column,
            policy=args.policy,
        )

        audit = audit_model_inputs(dataset)

        audit_path = write_model_input_audit(
            audit,
            args.summary_output,
        )

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Model-input preparation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    print(f"Model-input audit written to: {audit_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())