"""Tests for leakage-safe splitting and standardization."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from cyber_pca.preprocessing import (
    RawDataSplits,
    StandardizedDataSplits,
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.synthetic_data import (
    FEATURE_COLUMNS,
    generate_synthetic_network_data,
)


def test_default_split_sizes() -> None:
    dataset = generate_synthetic_network_data()

    splits = split_normal_calibration_test(dataset)

    assert isinstance(splits, RawDataSplits)
    assert splits.normal_fit.shape == (2400, 13)
    assert splits.normal_calibration.shape == (
        800,
        13,
    )
    assert splits.test.shape == (1800, 13)


def test_fit_and_calibration_are_normal_only() -> None:
    dataset = generate_synthetic_network_data()

    splits = split_normal_calibration_test(dataset)

    assert set(splits.normal_fit["is_anomaly"]) == {0}

    assert set(
        splits.normal_calibration["is_anomaly"]
    ) == {0}

    expected_test_counts = {
        "normal": 800,
        "port_scan": 250,
        "dos": 250,
        "brute_force": 250,
        "exfiltration": 250,
    }

    assert (
        splits.test["scenario"]
        .value_counts()
        .to_dict()
        == expected_test_counts
    )


def test_split_identifiers_are_disjoint_and_complete() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    splits = split_normal_calibration_test(dataset)

    fit_ids = set(splits.normal_fit["flow_id"])
    calibration_ids = set(
        splits.normal_calibration["flow_id"]
    )
    test_ids = set(splits.test["flow_id"])

    assert fit_ids.isdisjoint(calibration_ids)
    assert fit_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)

    combined_ids = (
        fit_ids
        | calibration_ids
        | test_ids
    )

    assert combined_ids == set(dataset["flow_id"])


def test_splitting_is_deterministic() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=73,
    )

    first = split_normal_calibration_test(
        dataset,
        random_seed=101,
    )

    second = split_normal_calibration_test(
        dataset,
        random_seed=101,
    )

    pd.testing.assert_frame_equal(
        first.normal_fit,
        second.normal_fit,
    )

    pd.testing.assert_frame_equal(
        first.normal_calibration,
        second.normal_calibration,
    )

    pd.testing.assert_frame_equal(
        first.test,
        second.test,
    )

def test_standardized_split_shapes_and_alignment() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    assert isinstance(
        standardized,
        StandardizedDataSplits,
    )

    assert standardized.normal_fit.shape == (
        72,
        10,
    )

    assert (
        standardized.normal_calibration.shape
        == (24, 10)
    )

    assert standardized.test.shape == (84, 10)

    assert tuple(
        standardized.normal_fit.columns
    ) == FEATURE_COLUMNS

    assert tuple(
        standardized.normal_calibration.columns
    ) == FEATURE_COLUMNS

    assert tuple(
        standardized.test.columns
    ) == FEATURE_COLUMNS

    assert standardized.normal_fit.index.name == (
        "flow_id"
    )

    np.testing.assert_array_equal(
        standardized.normal_fit.index.to_numpy(),
        raw_splits.normal_fit[
            "flow_id"
        ].to_numpy(),
    )

    np.testing.assert_array_equal(
        standardized.test.index.to_numpy(),
        raw_splits.test["flow_id"].to_numpy(),
    )


def test_fitting_features_have_zero_mean_and_unit_variance() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    fit_values = (
        standardized.normal_fit.to_numpy()
    )

    np.testing.assert_allclose(
        fit_values.mean(axis=0),
        np.zeros(len(FEATURE_COLUMNS)),
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        fit_values.std(axis=0, ddof=0),
        np.ones(len(FEATURE_COLUMNS)),
        atol=1.0e-12,
    )


def test_scaler_statistics_come_from_normal_fit_only() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    raw_fit = raw_splits.normal_fit.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        standardized.scaler.mean_,
        raw_fit.mean(axis=0),
    )

    np.testing.assert_allclose(
        standardized.scaler.var_,
        raw_fit.var(axis=0, ddof=0),
    )

def test_standardized_split_shapes_and_alignment() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    assert isinstance(
        standardized,
        StandardizedDataSplits,
    )

    assert standardized.normal_fit.shape == (
        72,
        10,
    )

    assert (
        standardized.normal_calibration.shape
        == (24, 10)
    )

    assert standardized.test.shape == (84, 10)

    assert tuple(
        standardized.normal_fit.columns
    ) == FEATURE_COLUMNS

    assert tuple(
        standardized.normal_calibration.columns
    ) == FEATURE_COLUMNS

    assert tuple(
        standardized.test.columns
    ) == FEATURE_COLUMNS

    assert standardized.normal_fit.index.name == (
        "flow_id"
    )

    np.testing.assert_array_equal(
        standardized.normal_fit.index.to_numpy(),
        raw_splits.normal_fit[
            "flow_id"
        ].to_numpy(),
    )

    np.testing.assert_array_equal(
        standardized.test.index.to_numpy(),
        raw_splits.test["flow_id"].to_numpy(),
    )


def test_fitting_features_have_zero_mean_and_unit_variance() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    fit_values = (
        standardized.normal_fit.to_numpy()
    )

    np.testing.assert_allclose(
        fit_values.mean(axis=0),
        np.zeros(len(FEATURE_COLUMNS)),
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        fit_values.std(axis=0, ddof=0),
        np.ones(len(FEATURE_COLUMNS)),
        atol=1.0e-12,
    )


def test_scaler_statistics_come_from_normal_fit_only() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    raw_fit = raw_splits.normal_fit.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        standardized.scaler.mean_,
        raw_fit.mean(axis=0),
    )

    np.testing.assert_allclose(
        standardized.scaler.var_,
        raw_fit.var(axis=0, ddof=0),
    )


def test_all_splits_use_the_same_scaler() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    raw_calibration = (
        raw_splits.normal_calibration.loc[
            :,
            FEATURE_COLUMNS,
        ].to_numpy(
            dtype=np.float64,
        )
    )

    expected_calibration = (
        raw_calibration
        - standardized.scaler.mean_
    ) / standardized.scaler.scale_

    np.testing.assert_allclose(
        standardized.normal_calibration.to_numpy(),
        expected_calibration,
    )

    raw_test = raw_splits.test.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
    )

    expected_test = (
        raw_test
        - standardized.scaler.mean_
    ) / standardized.scaler.scale_

    np.testing.assert_allclose(
        standardized.test.to_numpy(),
        expected_test,
    )


def test_test_extremes_do_not_change_scaler_statistics() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    baseline = standardize_splits(raw_splits)

    extreme_test = raw_splits.test.copy()

    extreme_test.loc[
        :,
        FEATURE_COLUMNS,
    ] = (
        extreme_test.loc[
            :,
            FEATURE_COLUMNS,
        ]
        + 1.0e12
    )

    altered_splits = RawDataSplits(
        normal_fit=raw_splits.normal_fit.copy(),
        normal_calibration=(
            raw_splits.normal_calibration.copy()
        ),
        test=extreme_test,
    )

    altered = standardize_splits(altered_splits)

    np.testing.assert_array_equal(
        baseline.scaler.mean_,
        altered.scaler.mean_,
    )

    np.testing.assert_array_equal(
        baseline.scaler.scale_,
        altered.scaler.scale_,
    )

    assert (
        np.abs(altered.test.to_numpy()).mean()
        > np.abs(baseline.test.to_numpy()).mean()
    )


def test_inverse_transform_recovers_raw_features() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized = standardize_splits(raw_splits)

    reconstructed = (
        standardized.scaler.inverse_transform(
            standardized.normal_calibration
        )
    )

    expected = (
        raw_splits.normal_calibration.loc[
            :,
            FEATURE_COLUMNS,
        ].to_numpy(
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        reconstructed,
        expected,
        rtol=1.0e-12,
        atol=1.0e-9,
    )

def test_splitting_does_not_mutate_input() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    original = dataset.copy(deep=True)

    split_normal_calibration_test(dataset)

    pd.testing.assert_frame_equal(
        dataset,
        original,
    )


def test_different_seed_changes_fit_membership() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=300,
        n_attack_per_type=30,
        random_seed=42,
    )

    first = split_normal_calibration_test(
        dataset,
        random_seed=101,
    )

    second = split_normal_calibration_test(
        dataset,
        random_seed=102,
    )

    first_ids = set(first.normal_fit["flow_id"])
    second_ids = set(second.normal_fit["flow_id"])

    assert first_ids != second_ids


def test_incorrect_column_order_is_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    reordered = dataset.loc[
        :,
        list(reversed(dataset.columns)),
    ]

    with pytest.raises(
        ValueError,
        match="OUTPUT_COLUMNS",
    ):
        split_normal_calibration_test(reordered)


def test_duplicate_flow_identifiers_are_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    dataset.loc[
        dataset.index[1],
        "flow_id",
    ] = dataset.loc[
        dataset.index[0],
        "flow_id",
    ]

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        split_normal_calibration_test(dataset)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_raw_features_are_rejected(
    invalid_value: float,
) -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    dataset.loc[
        dataset.index[0],
        "duration_ms",
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        split_normal_calibration_test(dataset)


@pytest.mark.parametrize(
    (
        "fit_fraction",
        "calibration_fraction",
        "expected_exception",
    ),
    [
        (0.0, 0.20, ValueError),
        (1.0, 0.20, ValueError),
        ("0.60", 0.20, TypeError),
        (True, 0.20, TypeError),
        (0.60, 0.0, ValueError),
        (0.60, 1.0, ValueError),
        (0.80, 0.20, ValueError),
    ],
)
def test_invalid_split_fractions_are_rejected(
    fit_fraction: object,
    calibration_fraction: object,
    expected_exception: type[Exception],
) -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    with pytest.raises(expected_exception):
        split_normal_calibration_test(
            dataset,
            normal_fit_fraction=fit_fraction,
            normal_calibration_fraction=(
                calibration_fraction
            ),
        )


@pytest.mark.parametrize(
    ("invalid_seed", "expected_exception"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_invalid_split_seeds_are_rejected(
    invalid_seed: object,
    expected_exception: type[Exception],
) -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    with pytest.raises(expected_exception):
        split_normal_calibration_test(
            dataset,
            random_seed=invalid_seed,
        )


@pytest.mark.parametrize(
    "invalid_shuffle",
    [
        1,
        "yes",
        None,
    ],
)
def test_invalid_test_shuffle_values_are_rejected(
    invalid_shuffle: object,
) -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    with pytest.raises(TypeError):
        split_normal_calibration_test(
            dataset,
            shuffle_test=invalid_shuffle,
        )


def test_split_rejects_empty_normal_partition() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=2,
        n_attack_per_type=1,
        random_seed=42,
    )

    with pytest.raises(
        ValueError,
        match="nonempty",
    ):
        split_normal_calibration_test(dataset)


def test_calibration_extremes_do_not_change_scaler_statistics() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    baseline = standardize_splits(raw_splits)

    extreme_calibration = (
        raw_splits.normal_calibration.copy()
    )

    extreme_calibration.loc[
        :,
        FEATURE_COLUMNS,
    ] = (
        extreme_calibration.loc[
            :,
            FEATURE_COLUMNS,
        ]
        + 1.0e12
    )

    altered_splits = RawDataSplits(
        normal_fit=raw_splits.normal_fit.copy(),
        normal_calibration=extreme_calibration,
        test=raw_splits.test.copy(),
    )

    altered = standardize_splits(altered_splits)

    np.testing.assert_array_equal(
        baseline.scaler.mean_,
        altered.scaler.mean_,
    )

    np.testing.assert_array_equal(
        baseline.scaler.scale_,
        altered.scaler.scale_,
    )


def test_zero_variance_fitting_feature_is_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    constant_fit = raw_splits.normal_fit.copy()
    constant_fit["failed_logins"] = 0.0

    altered_splits = RawDataSplits(
        normal_fit=constant_fit,
        normal_calibration=(
            raw_splits.normal_calibration.copy()
        ),
        test=raw_splits.test.copy(),
    )

    with pytest.raises(
        ValueError,
        match="zero variance",
    ):
        standardize_splits(altered_splits)


def test_overlapping_split_identifiers_are_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    overlapping_calibration = (
        raw_splits.normal_calibration.copy()
    )

    overlapping_calibration.loc[
        overlapping_calibration.index[0],
        "flow_id",
    ] = raw_splits.normal_fit.loc[
        raw_splits.normal_fit.index[0],
        "flow_id",
    ]

    altered_splits = RawDataSplits(
        normal_fit=raw_splits.normal_fit.copy(),
        normal_calibration=(
            overlapping_calibration
        ),
        test=raw_splits.test.copy(),
    )

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        standardize_splits(altered_splits)


def test_nonfinite_calibration_feature_is_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=15,
        random_seed=42,
    )

    raw_splits = split_normal_calibration_test(
        dataset
    )

    invalid_calibration = (
        raw_splits.normal_calibration.copy()
    )

    invalid_calibration.loc[
        invalid_calibration.index[0],
        "bytes_out",
    ] = np.inf

    altered_splits = RawDataSplits(
        normal_fit=raw_splits.normal_fit.copy(),
        normal_calibration=invalid_calibration,
        test=raw_splits.test.copy(),
    )

    with pytest.raises(
        ValueError,
        match="nonfinite",
    ):
        standardize_splits(altered_splits)


def test_invalid_normal_scenario_is_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    normal_index = dataset.index[
        dataset["is_anomaly"] == 0
    ][0]

    dataset.loc[
        normal_index,
        "scenario",
    ] = "port_scan"

    with pytest.raises(
        ValueError,
        match="label-0",
    ):
        split_normal_calibration_test(dataset)


def test_unknown_attack_scenario_is_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    attack_index = dataset.index[
        dataset["is_anomaly"] == 1
    ][0]

    dataset.loc[
        attack_index,
        "scenario",
    ] = "unknown_attack"

    with pytest.raises(
        ValueError,
        match="ATTACK_TYPES",
    ):
        split_normal_calibration_test(dataset)


def test_invalid_anomaly_label_is_rejected() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=30,
        n_attack_per_type=5,
    )

    dataset.loc[
        dataset.index[0],
        "is_anomaly",
    ] = 2

    with pytest.raises(
        ValueError,
        match="labels 0 and 1",
    ):
        split_normal_calibration_test(dataset)
