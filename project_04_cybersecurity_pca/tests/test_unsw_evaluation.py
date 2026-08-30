"""Tests for official UNSW-NB15 label alignment and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cyber_pca.evaluation import (
    evaluate_binary_predictions,
)
from cyber_pca.unsw_evaluation import (
    align_unsw_evaluation_data,
    evaluate_unsw_attack_categories,
)


ATTACK_CATEGORY_ORDER = (
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
)


def _evaluation_fixture(
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    raw_test = pd.DataFrame(
        {
            "id": np.arange(
                1,
                11,
                dtype=np.int64,
            ),
            "label": np.asarray(
                [0] + [1] * 9,
                dtype=np.int8,
            ),
            "attack_cat": (
                ATTACK_CATEGORY_ORDER
            ),
        }
    )

    raw_test = (
        raw_test.sample(
            frac=1.0,
            random_state=42,
        )
        .reset_index(drop=True)
    )

    flow_ids = pd.Index(
        [
            f"unsw_testing:{value}"
            for value in range(1, 11)
        ],
        name="flow_id",
    )

    reconstruction_errors = pd.Series(
        np.linspace(
            0.05,
            0.95,
            num=10,
            dtype=np.float64,
        ),
        index=flow_ids,
        name="reconstruction_error",
    )

    predictions = pd.Series(
        np.asarray(
            [
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ],
            dtype=np.int8,
        ),
        index=flow_ids,
        name="predicted_anomaly",
    )

    return (
        raw_test,
        reconstruction_errors,
        predictions,
    )


def test_aligns_hidden_labels_by_partition_id(
) -> None:
    (
        raw_test,
        reconstruction_errors,
        predictions,
    ) = _evaluation_fixture()

    raw_before = raw_test.copy(
        deep=True
    )
    errors_before = (
        reconstruction_errors.copy(
            deep=True
        )
    )
    predictions_before = (
        predictions.copy(deep=True)
    )

    evaluation_data = (
        align_unsw_evaluation_data(
            raw_test,
            reconstruction_errors,
            predictions,
        )
    )

    assert tuple(
        evaluation_data.columns
    ) == (
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    )

    assert evaluation_data.index.name == (
        "flow_id"
    )

    assert evaluation_data.index.equals(
        reconstruction_errors.index
    )

    assert evaluation_data[
        "true_anomaly"
    ].tolist() == (
        [0] + [1] * 9
    )

    assert evaluation_data[
        "scenario"
    ].tolist() == list(
        ATTACK_CATEGORY_ORDER
    )

    assert evaluation_data[
        "predicted_anomaly"
    ].tolist() == predictions.tolist()

    np.testing.assert_allclose(
        evaluation_data[
            "reconstruction_error"
        ].to_numpy(
            dtype=np.float64,
        ),
        reconstruction_errors.to_numpy(
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )

    binary_result = (
        evaluate_binary_predictions(
            evaluation_data
        )
    )

    assert binary_result.total == 10
    assert binary_result.normal_support == 1
    assert binary_result.anomaly_support == 9

    pd.testing.assert_frame_equal(
        raw_test,
        raw_before,
    )
    pd.testing.assert_series_equal(
        reconstruction_errors,
        errors_before,
    )
    pd.testing.assert_series_equal(
        predictions,
        predictions_before,
    )


def test_evaluates_official_attack_categories(
) -> None:
    (
        raw_test,
        reconstruction_errors,
        predictions,
    ) = _evaluation_fixture()

    evaluation_data = (
        align_unsw_evaluation_data(
            raw_test,
            reconstruction_errors,
            predictions,
        )
    )

    category_result = (
        evaluate_unsw_attack_categories(
            evaluation_data
        )
    )

    assert tuple(
        category_result.columns
    ) == (
        "attack_category",
        "true_label",
        "observations",
        "predicted_normal",
        "predicted_anomaly",
        "predicted_anomaly_rate",
        "mean_reconstruction_error",
        "median_reconstruction_error",
        "maximum_reconstruction_error",
    )

    assert category_result[
        "attack_category"
    ].tolist() == list(
        ATTACK_CATEGORY_ORDER
    )

    assert category_result[
        "true_label"
    ].tolist() == (
        [0] + [1] * 9
    )

    assert category_result[
        "observations"
    ].tolist() == [1] * 10

    assert category_result[
        "predicted_anomaly"
    ].tolist() == predictions.tolist()

    assert category_result[
        "predicted_normal"
    ].tolist() == [
        1 - value
        for value in predictions.tolist()
    ]

    np.testing.assert_allclose(
        category_result[
            "predicted_anomaly_rate"
        ].to_numpy(
            dtype=np.float64,
        ),
        predictions.to_numpy(
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )
