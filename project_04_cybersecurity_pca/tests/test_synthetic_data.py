"""Tests for deterministic synthetic cybersecurity data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyber_pca.synthetic_data import (
    ATTACK_TYPES,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    generate_synthetic_network_data,
)


EXPECTED_FEATURE_COLUMNS = (
    "duration_ms",
    "packets_in",
    "packets_out",
    "bytes_in",
    "bytes_out",
    "syn_count",
    "ack_count",
    "connection_rate",
    "unique_dest_ports",
    "failed_logins",
)

EXPECTED_ATTACK_TYPES = (
    "port_scan",
    "dos",
    "brute_force",
    "exfiltration",
)

EXPECTED_OUTPUT_COLUMNS = (
    "flow_id",
    *EXPECTED_FEATURE_COLUMNS,
    "is_anomaly",
    "scenario",
)


def test_synthetic_schema_constants() -> None:
    assert FEATURE_COLUMNS == EXPECTED_FEATURE_COLUMNS
    assert ATTACK_TYPES == EXPECTED_ATTACK_TYPES
    assert OUTPUT_COLUMNS == EXPECTED_OUTPUT_COLUMNS


def test_default_dataset_has_expected_shape() -> None:
    dataset = generate_synthetic_network_data()

    assert dataset.shape == (5000, 13)
    assert tuple(dataset.columns) == EXPECTED_OUTPUT_COLUMNS


def test_default_scenario_counts_and_labels() -> None:
    dataset = generate_synthetic_network_data()

    expected_counts = {
        "normal": 4000,
        "port_scan": 250,
        "dos": 250,
        "brute_force": 250,
        "exfiltration": 250,
    }

    assert dataset["scenario"].value_counts().to_dict() == (
        expected_counts
    )

    expected_labels = {
        "normal": 0,
        "port_scan": 1,
        "dos": 1,
        "brute_force": 1,
        "exfiltration": 1,
    }

    for scenario, expected_label in expected_labels.items():
        scenario_labels = set(
            dataset.loc[
                dataset["scenario"] == scenario,
                "is_anomaly",
            ]
        )

        assert scenario_labels == {expected_label}

def test_features_are_float64_finite_and_nonnegative() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=600,
        n_attack_per_type=100,
        random_seed=42,
    )

    feature_frame = dataset.loc[:, FEATURE_COLUMNS]
    feature_values = feature_frame.to_numpy()

    assert all(
        dtype == np.dtype(np.float64)
        for dtype in feature_frame.dtypes
    )
    assert np.all(np.isfinite(feature_values))
    assert np.all(feature_values >= 0.0)
    assert np.all(
        feature_frame.std(axis=0).to_numpy() > 0.0
    )


def test_flow_identifiers_are_unique_and_sequential() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=30,
        random_seed=42,
    )

    assert dataset["flow_id"].dtype == np.dtype(
        np.int64
    )
    assert dataset["flow_id"].is_unique

    np.testing.assert_array_equal(
        dataset["flow_id"].to_numpy(),
        np.arange(
            len(dataset),
            dtype=np.int64,
        ),
    )


def test_same_seed_produces_identical_dataset() -> None:
    first = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=30,
        random_seed=101,
    )

    second = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=30,
        random_seed=101,
    )

    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_feature_values() -> None:
    first = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=30,
        random_seed=101,
    )

    second = generate_synthetic_network_data(
        n_normal=120,
        n_attack_per_type=30,
        random_seed=102,
    )

    first_features = first.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy()

    second_features = second.loc[
        :,
        FEATURE_COLUMNS,
    ].to_numpy()

    assert not np.array_equal(
        first_features,
        second_features,
    )


def test_attack_scenarios_do_not_contain_duplicate_rows() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=500,
        n_attack_per_type=100,
        random_seed=42,
    )

    for attack_type in ATTACK_TYPES:
        attack_features = dataset.loc[
            dataset["scenario"] == attack_type,
            FEATURE_COLUMNS,
        ]

        assert not attack_features.duplicated().any()


def test_attack_signatures_are_statistically_visible() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=1500,
        n_attack_per_type=300,
        random_seed=42,
    )

    medians = dataset.groupby("scenario")[
        list(FEATURE_COLUMNS)
    ].median()

    packet_totals = (
        dataset["packets_in"]
        + dataset["packets_out"]
    )

    packet_medians = packet_totals.groupby(
        dataset["scenario"]
    ).median()

    outbound_ratios = dataset["bytes_out"] / (
        dataset["bytes_in"] + 1.0
    )

    ratio_medians = outbound_ratios.groupby(
        dataset["scenario"]
    ).median()

    assert (
        medians.loc[
            "port_scan",
            "unique_dest_ports",
        ]
        > 10.0
        * medians.loc[
            "normal",
            "unique_dest_ports",
        ]
    )

    assert (
        medians.loc["port_scan", "syn_count"]
        > 5.0 * medians.loc["normal", "syn_count"]
    )

    assert (
        medians.loc["port_scan", "bytes_out"]
        < 0.5 * medians.loc["normal", "bytes_out"]
    )

    assert (
        medians.loc["dos", "connection_rate"]
        > 8.0
        * medians.loc["normal", "connection_rate"]
    )

    assert (
        packet_medians.loc["dos"]
        > 5.0 * packet_medians.loc["normal"]
    )

    assert (
        medians.loc["brute_force", "failed_logins"]
        > medians.loc["normal", "failed_logins"]
        + 10.0
    )

    assert (
        medians.loc["brute_force", "connection_rate"]
        > 3.0
        * medians.loc["normal", "connection_rate"]
    )

    assert (
        medians.loc[
            "brute_force",
            "unique_dest_ports",
        ]
        < 0.1
        * medians.loc[
            "port_scan",
            "unique_dest_ports",
        ]
    )

    assert (
        medians.loc["exfiltration", "bytes_out"]
        > 8.0 * medians.loc["normal", "bytes_out"]
    )

    assert (
        ratio_medians.loc["exfiltration"]
        > 8.0 * ratio_medians.loc["normal"]
    )


def test_unshuffled_dataset_has_documented_order() -> None:
    dataset = generate_synthetic_network_data(
        n_normal=3,
        n_attack_per_type=2,
        random_seed=42,
        shuffle=False,
    )

    expected_scenarios = [
        "normal",
        "normal",
        "normal",
        "port_scan",
        "port_scan",
        "dos",
        "dos",
        "brute_force",
        "brute_force",
        "exfiltration",
        "exfiltration",
    ]

    assert dataset["scenario"].tolist() == (
        expected_scenarios
    )


@pytest.mark.parametrize(
    (
        "parameter_name",
        "invalid_value",
        "expected_exception",
    ),
    [
        ("n_normal", 0, ValueError),
        ("n_normal", -1, ValueError),
        ("n_normal", 1.5, TypeError),
        ("n_normal", True, TypeError),
        ("n_attack_per_type", 0, ValueError),
        ("n_attack_per_type", -1, ValueError),
        ("n_attack_per_type", 1.5, TypeError),
        ("n_attack_per_type", True, TypeError),
    ],
)
def test_invalid_observation_counts_are_rejected(
    parameter_name: str,
    invalid_value: object,
    expected_exception: type[Exception],
) -> None:
    arguments = {
        "n_normal": 10,
        "n_attack_per_type": 3,
        parameter_name: invalid_value,
    }

    with pytest.raises(expected_exception):
        generate_synthetic_network_data(**arguments)


@pytest.mark.parametrize(
    ("invalid_seed", "expected_exception"),
    [
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ],
)
def test_invalid_random_seeds_are_rejected(
    invalid_seed: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        generate_synthetic_network_data(
            n_normal=10,
            n_attack_per_type=3,
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
def test_invalid_shuffle_values_are_rejected(
    invalid_shuffle: object,
) -> None:
    with pytest.raises(TypeError):
        generate_synthetic_network_data(
            n_normal=10,
            n_attack_per_type=3,
            shuffle=invalid_shuffle,
        )
