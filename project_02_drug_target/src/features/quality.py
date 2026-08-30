"""Audit feature quality using only the frozen training partition."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.dataset import ModelDataset, POLICIES, load_train_test_data


AUDIT_PARTITION = "train"
DEFAULT_HIGH_CORRELATION_THRESHOLD = 0.95

ZERO_VARIANCE_ACTION = (
    "Future model pipelines will fit VarianceThreshold(threshold=0.0) "
    "on their training data only."
)

CORRELATION_ACTION = (
    "Retain correlated descriptors for now and document them; do not apply "
    "automatic correlation pruning or label-driven feature selection."
)


class FeatureQualityError(ValueError):
    """Raised when a training-feature quality audit cannot be completed."""


@dataclass(frozen=True)
class FeatureStatistics:
    """Descriptive statistics for one training feature."""

    feature: str
    unique_value_count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    variance: float


@dataclass(frozen=True)
class HighCorrelationPair:
    """One high absolute Pearson-correlation pair from training data."""

    feature_a: str
    feature_b: str
    pearson_correlation: float
    absolute_pearson_correlation: float


@dataclass(frozen=True)
class FeatureQualityAudit:
    """Version-controlled, train-only feature-quality evidence."""

    policy: str
    label_column: str
    audit_partition: str
    audit_pair_count: int
    input_feature_count: int
    usable_feature_count: int
    feature_columns: tuple[str, ...]
    retained_feature_columns: tuple[str, ...]
    missing_feature_value_count: int
    nonfinite_feature_value_count: int
    feature_statistics: tuple[FeatureStatistics, ...]
    zero_variance_feature_count: int
    zero_variance_features: tuple[str, ...]
    zero_variance_action: str
    high_correlation_threshold: float
    high_correlation_pair_count: int
    high_correlation_pairs: tuple[HighCorrelationPair, ...]
    correlation_action: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record."""
        return asdict(self)


def _validated_threshold(value: float) -> float:
    """Return a valid absolute Pearson-correlation screening threshold."""
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise FeatureQualityError(
            "high_correlation_threshold must be numeric."
        ) from error

    if not 0.0 < threshold <= 1.0:
        raise FeatureQualityError(
            "high_correlation_threshold must be in the interval (0, 1]."
        )

    return threshold


def _validated_training_values(dataset: ModelDataset) -> pd.DataFrame:
    """Return finite numeric training features under the frozen contract."""
    if dataset.X_train.empty:
        raise FeatureQualityError("Training feature matrix is empty.")

    feature_columns = tuple(dataset.feature_columns)

    if not feature_columns:
        raise FeatureQualityError("Feature-column contract is empty.")

    if len(set(feature_columns)) != len(feature_columns):
        raise FeatureQualityError(
            "Feature-column contract contains duplicate names."
        )

    if dataset.X_train.columns.duplicated().any():
        raise FeatureQualityError(
            "Training feature matrix contains duplicate column names."
        )

    if tuple(dataset.X_train.columns) != feature_columns:
        raise FeatureQualityError(
            "Training feature columns do not match the frozen feature contract."
        )

    try:
        values = dataset.X_train.loc[:, list(feature_columns)].apply(
            pd.to_numeric,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise FeatureQualityError(
            "Training feature columns must be numeric."
        ) from error

    missing_count = int(values.isna().sum().sum())

    if missing_count:
        raise FeatureQualityError(
            f"Training feature columns contain {missing_count} missing values."
        )

    numeric_values = values.to_numpy(dtype=float)
    nonfinite_count = int((~np.isfinite(numeric_values)).sum())

    if nonfinite_count:
        raise FeatureQualityError(
            "Training feature columns contain "
            f"{nonfinite_count} non-finite values."
        )

    return values.astype(float)


def _feature_statistics(
    training_values: pd.DataFrame,
) -> tuple[FeatureStatistics, ...]:
    """Calculate deterministic descriptive statistics for training features."""
    variances = training_values.var(axis=0, ddof=0)
    standard_deviations = training_values.std(axis=0, ddof=0)

    return tuple(
        FeatureStatistics(
            feature=column,
            unique_value_count=int(training_values[column].nunique()),
            minimum=float(training_values[column].min()),
            maximum=float(training_values[column].max()),
            mean=float(training_values[column].mean()),
            standard_deviation=float(standard_deviations[column]),
            variance=float(variances[column]),
        )
        for column in training_values.columns
    )


def _high_correlation_pairs(
    training_values: pd.DataFrame,
    feature_columns: tuple[str, ...],
    threshold: float,
) -> tuple[HighCorrelationPair, ...]:
    """Return all training-only feature pairs above an absolute threshold."""
    if len(feature_columns) < 2:
        return ()

    correlations = training_values.loc[
        :,
        list(feature_columns),
    ].corr(method="pearson")

    pairs: list[HighCorrelationPair] = []

    for left_position, left_feature in enumerate(feature_columns[:-1]):
        for right_feature in feature_columns[left_position + 1 :]:
            correlation = float(
                correlations.loc[left_feature, right_feature]
            )

            if not np.isfinite(correlation):
                continue

            absolute_correlation = abs(correlation)

            if absolute_correlation >= threshold:
                pairs.append(
                    HighCorrelationPair(
                        feature_a=left_feature,
                        feature_b=right_feature,
                        pearson_correlation=correlation,
                        absolute_pearson_correlation=absolute_correlation,
                    )
                )

    pairs.sort(
        key=lambda pair: (
            -pair.absolute_pearson_correlation,
            pair.feature_a,
            pair.feature_b,
        )
    )

    return tuple(pairs)


def audit_training_feature_quality(
    dataset: ModelDataset,
    *,
    high_correlation_threshold: float = (
        DEFAULT_HIGH_CORRELATION_THRESHOLD
    ),
) -> FeatureQualityAudit:
    """Audit only the frozen training feature matrix.

    Labels, test features, and test labels are not used to calculate quality
    statistics, zero-variance features, or correlation pairs.
    """
    threshold = _validated_threshold(high_correlation_threshold)
    training_values = _validated_training_values(dataset)

    statistics = _feature_statistics(training_values)

    zero_variance_features = tuple(
        statistic.feature
        for statistic in statistics
        if statistic.variance == 0.0
    )

    zero_variance_set = set(zero_variance_features)

    retained_feature_columns = tuple(
        column
        for column in dataset.feature_columns
        if column not in zero_variance_set
    )

    correlation_pairs = _high_correlation_pairs(
        training_values,
        retained_feature_columns,
        threshold,
    )

    return FeatureQualityAudit(
        policy=dataset.policy,
        label_column=dataset.label_column,
        audit_partition=AUDIT_PARTITION,
        audit_pair_count=int(len(training_values)),
        input_feature_count=int(len(dataset.feature_columns)),
        usable_feature_count=int(len(retained_feature_columns)),
        feature_columns=tuple(dataset.feature_columns),
        retained_feature_columns=retained_feature_columns,
        missing_feature_value_count=0,
        nonfinite_feature_value_count=0,
        feature_statistics=statistics,
        zero_variance_feature_count=int(len(zero_variance_features)),
        zero_variance_features=zero_variance_features,
        zero_variance_action=ZERO_VARIANCE_ACTION,
        high_correlation_threshold=threshold,
        high_correlation_pair_count=int(len(correlation_pairs)),
        high_correlation_pairs=correlation_pairs,
        correlation_action=CORRELATION_ACTION,
    )


def write_feature_quality_audit(
    audit: FeatureQualityAudit,
    output_path: str | Path,
) -> Path:
    """Write compact, version-controlled feature-quality evidence."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return destination


def main(argv: list[str] | None = None) -> int:
    """Run a train-only feature-quality audit from local Davis artifacts."""
    parser = argparse.ArgumentParser(
        description="Audit frozen Davis training features without test leakage."
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
        help="Pre-specified binary label column used only to load partitions.",
    )

    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="cold_drug",
        help="Frozen split policy to audit.",
    )

    parser.add_argument(
        "--high-correlation-threshold",
        type=float,
        default=DEFAULT_HIGH_CORRELATION_THRESHOLD,
        help="Report training-only Pearson pairs with absolute correlation "
        "at or above this value.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_training_feature_quality.json"),
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

        audit = audit_training_feature_quality(
            dataset,
            high_correlation_threshold=args.high_correlation_threshold,
        )

        output_path = write_feature_quality_audit(
            audit,
            args.summary_output,
        )

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Feature-quality audit failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    print(f"Feature-quality audit written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())