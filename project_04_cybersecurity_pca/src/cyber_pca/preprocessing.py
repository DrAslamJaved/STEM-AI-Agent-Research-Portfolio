"""Leakage-safe splitting and standardization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from cyber_pca.synthetic_data import (
    ATTACK_TYPES,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
)


DEFAULT_NORMAL_FIT_FRACTION = 0.60
DEFAULT_NORMAL_CALIBRATION_FRACTION = 0.20
DEFAULT_SPLIT_RANDOM_SEED = 42


@dataclass(frozen=True)
class RawDataSplits:
    """Raw fitting, calibration, and test partitions."""

    normal_fit: pd.DataFrame
    normal_calibration: pd.DataFrame
    test: pd.DataFrame

@dataclass(frozen=True)
class StandardizedDataSplits:
    """Standardized feature partitions and fitted scaler."""

    normal_fit: pd.DataFrame
    normal_calibration: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler

def _validate_fraction(
    value: object,
    *,
    name: str,
) -> float:
    """Validate a split fraction."""

    if isinstance(value, bool) or not isinstance(
        value,
        Real,
    ):
        raise TypeError(f"{name} must be a real number.")

    float_value = float(value)

    if not np.isfinite(float_value):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 < float_value < 1.0:
        raise ValueError(
            f"{name} must be strictly between 0 and 1."
        )

    return float_value


def _validate_random_seed(value: object) -> int:
    """Validate a nonnegative integer random seed."""

    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise TypeError("random_seed must be an integer.")

    integer_value = int(value)

    if integer_value < 0:
        raise ValueError(
            "random_seed must be nonnegative."
        )

    return integer_value


def _validate_dataset(dataset: object) -> pd.DataFrame:
    """Validate the raw cybersecurity dataset contract."""

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame.")

    if dataset.empty:
        raise ValueError("dataset must not be empty.")

    if tuple(dataset.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "dataset columns must exactly match "
            "OUTPUT_COLUMNS."
        )

    if dataset["flow_id"].isna().any():
        raise ValueError(
            "flow_id must not contain missing values."
        )

    if dataset["flow_id"].duplicated().any():
        raise ValueError(
            "flow_id values must be unique."
        )

    labels = set(dataset["is_anomaly"].tolist())

    if labels != {0, 1}:
        raise ValueError(
            "is_anomaly must contain both labels 0 and 1."
        )

    normal_rows = dataset["is_anomaly"] == 0
    attack_rows = dataset["is_anomaly"] == 1

    if not (
        dataset.loc[
            normal_rows,
            "scenario",
        ]
        == "normal"
    ).all():
        raise ValueError(
            "Every label-0 observation must have "
            "scenario 'normal'."
        )

    attack_scenarios = set(
        dataset.loc[
            attack_rows,
            "scenario",
        ]
    )

    if not attack_scenarios.issubset(
        set(ATTACK_TYPES)
    ):
        raise ValueError(
            "Attack scenarios must belong to "
            "ATTACK_TYPES."
        )

    feature_values = dataset.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.all(np.isfinite(feature_values)):
        raise ValueError(
            "Model features must contain finite values."
        )

    return dataset


def split_normal_calibration_test(
    dataset: pd.DataFrame,
    normal_fit_fraction: float = (
        DEFAULT_NORMAL_FIT_FRACTION
    ),
    normal_calibration_fraction: float = (
        DEFAULT_NORMAL_CALIBRATION_FRACTION
    ),
    random_seed: int = DEFAULT_SPLIT_RANDOM_SEED,
    *,
    shuffle_test: bool = True,
) -> RawDataSplits:
    """Split data without exposing attacks to fitting.

    Normal observations are divided among fitting,
    calibration, and test partitions. Every attack
    observation is assigned to the test partition.
    """

    validated_dataset = _validate_dataset(dataset)

    fit_fraction = _validate_fraction(
        normal_fit_fraction,
        name="normal_fit_fraction",
    )

    calibration_fraction = _validate_fraction(
        normal_calibration_fraction,
        name="normal_calibration_fraction",
    )

    if fit_fraction + calibration_fraction >= 1.0:
        raise ValueError(
            "Fitting and calibration fractions must "
            "sum to less than 1."
        )

    validated_seed = _validate_random_seed(
        random_seed
    )

    if not isinstance(shuffle_test, bool):
        raise TypeError(
            "shuffle_test must be a boolean."
        )

    normal = validated_dataset.loc[
        validated_dataset["is_anomaly"] == 0
    ].copy()

    attacks = validated_dataset.loc[
        validated_dataset["is_anomaly"] == 1
    ].copy()

    normal_count = len(normal)

    fit_count = int(
        np.floor(normal_count * fit_fraction)
    )

    calibration_count = int(
        np.floor(
            normal_count * calibration_fraction
        )
    )

    normal_test_count = (
        normal_count
        - fit_count
        - calibration_count
    )

    if min(
        fit_count,
        calibration_count,
        normal_test_count,
    ) <= 0:
        raise ValueError(
            "Split fractions must produce nonempty "
            "normal fitting, calibration, and test "
            "partitions."
        )

    rng = np.random.default_rng(validated_seed)

    normal_permutation = rng.permutation(
        normal_count
    )

    shuffled_normal = normal.iloc[
        normal_permutation
    ].reset_index(drop=True)

    normal_fit = shuffled_normal.iloc[
        :fit_count
    ].copy()

    normal_calibration = shuffled_normal.iloc[
        fit_count : fit_count + calibration_count
    ].reset_index(drop=True)

    normal_test = shuffled_normal.iloc[
        fit_count + calibration_count :
    ].reset_index(drop=True)

    normal_fit = normal_fit.reset_index(drop=True)

    test = pd.concat(
        [
            normal_test,
            attacks.reset_index(drop=True),
        ],
        axis=0,
        ignore_index=True,
    )

    if shuffle_test:
        test_permutation = rng.permutation(len(test))
        test = test.iloc[
            test_permutation
        ].reset_index(drop=True)

    return RawDataSplits(
        normal_fit=normal_fit,
        normal_calibration=normal_calibration,
        test=test.copy(),
    )

def _validate_raw_splits(
    splits: object,
) -> RawDataSplits:
    """Validate a raw partition container."""

    if not isinstance(splits, RawDataSplits):
        raise TypeError(
            "splits must be a RawDataSplits instance."
        )

    named_frames = (
        ("normal_fit", splits.normal_fit),
        (
            "normal_calibration",
            splits.normal_calibration,
        ),
        ("test", splits.test),
    )

    identifier_sets: dict[str, set[object]] = {}

    for name, frame in named_frames:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

        if frame.empty:
            raise ValueError(
                f"{name} must not be empty."
            )

        if tuple(frame.columns) != OUTPUT_COLUMNS:
            raise ValueError(
                f"{name} columns must exactly match "
                "OUTPUT_COLUMNS."
            )

        if frame["flow_id"].isna().any():
            raise ValueError(
                f"{name} contains missing flow IDs."
            )

        if frame["flow_id"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate flow IDs."
            )

        feature_values = frame.loc[
            :,
            FEATURE_COLUMNS,
        ].to_numpy(
            dtype=np.float64,
        )

        if not np.all(np.isfinite(feature_values)):
            raise ValueError(
                f"{name} contains nonfinite features."
            )

        identifier_sets[name] = set(
            frame["flow_id"]
        )

    if set(splits.normal_fit["is_anomaly"]) != {0}:
        raise ValueError(
            "normal_fit must contain normal labels only."
        )

    if (
        set(
            splits.normal_calibration["is_anomaly"]
        )
        != {0}
    ):
        raise ValueError(
            "normal_calibration must contain normal "
            "labels only."
        )

    if set(splits.test["is_anomaly"]) != {0, 1}:
        raise ValueError(
            "test must contain normal and anomaly labels."
        )

    fit_ids = identifier_sets["normal_fit"]
    calibration_ids = identifier_sets[
        "normal_calibration"
    ]
    test_ids = identifier_sets["test"]

    if not fit_ids.isdisjoint(calibration_ids):
        raise ValueError(
            "Fitting and calibration IDs overlap."
        )

    if not fit_ids.isdisjoint(test_ids):
        raise ValueError(
            "Fitting and test IDs overlap."
        )

    if not calibration_ids.isdisjoint(test_ids):
        raise ValueError(
            "Calibration and test IDs overlap."
        )

    return splits


def _standardized_frame(
    values: np.ndarray,
    raw_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create an aligned standardized feature frame."""

    return pd.DataFrame(
        values,
        columns=FEATURE_COLUMNS,
        index=pd.Index(
            raw_frame["flow_id"].to_numpy(
                copy=True
            ),
            name="flow_id",
        ),
        dtype=np.float64,
    )


def standardize_splits(
    splits: RawDataSplits,
) -> StandardizedDataSplits:
    """Fit scaling on normal fitting features only."""

    validated_splits = _validate_raw_splits(splits)

    fit_values = validated_splits.normal_fit.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
    )

    fit_variances = fit_values.var(
        axis=0,
        ddof=0,
    )

    zero_variance_indices = np.flatnonzero(
        fit_variances <= 0.0
    )

    if zero_variance_indices.size > 0:
        zero_variance_features = [
            FEATURE_COLUMNS[index]
            for index in zero_variance_indices
        ]

        raise ValueError(
            "Normal fitting features contain zero "
            "variance: "
            + ", ".join(zero_variance_features)
        )

    scaler = StandardScaler(
        copy=True,
        with_mean=True,
        with_std=True,
    )

    standardized_fit_values = scaler.fit_transform(
        fit_values
    )

    calibration_values = (
        validated_splits.normal_calibration.loc[
            :,
            FEATURE_COLUMNS,
        ].to_numpy(
            dtype=np.float64,
        )
    )

    test_values = validated_splits.test.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
    )

    standardized_calibration_values = (
        scaler.transform(calibration_values)
    )

    standardized_test_values = scaler.transform(
        test_values
    )

    return StandardizedDataSplits(
        normal_fit=_standardized_frame(
            standardized_fit_values,
            validated_splits.normal_fit,
        ),
        normal_calibration=_standardized_frame(
            standardized_calibration_values,
            validated_splits.normal_calibration,
        ),
        test=_standardized_frame(
            standardized_test_values,
            validated_splits.test,
        ),
        scaler=scaler,
    )
