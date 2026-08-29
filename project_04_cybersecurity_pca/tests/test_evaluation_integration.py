"""Integration tests for frozen synthetic evaluation."""

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cyber_pca import (
    align_evaluation_data,
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    evaluate_binary_predictions,
    evaluate_scenarios,
    fit_normal_pca,
    generate_synthetic_network_data,
    predict_anomalies,
    split_normal_calibration_test,
    standardize_splits,
)


@pytest.fixture(scope="module")
def phase_six_evidence() -> SimpleNamespace:
    dataset = generate_synthetic_network_data()

    raw_splits = split_normal_calibration_test(
        dataset
    )

    standardized_splits = standardize_splits(
        raw_splits
    )

    fit_result = fit_normal_pca(
        standardized_splits
    )

    error_splits = compute_reconstruction_errors(
        standardized_splits,
        fit_result,
    )

    threshold_result = calibrate_anomaly_threshold(
        error_splits
    )

    components_before = (
        fit_result.model.components_.copy()
    )
    mean_before = fit_result.model.mean_.copy()
    threshold_before = threshold_result.threshold
    errors_before = error_splits.test.copy(
        deep=True
    )

    predictions = predict_anomalies(
        error_splits.test,
        threshold_result,
    )

    predictions_before = predictions.copy(
        deep=True
    )
    raw_test_before = raw_splits.test.copy(
        deep=True
    )

    evaluation_data = align_evaluation_data(
        raw_splits.test,
        error_splits.test,
        predictions,
    )

    binary_result = evaluate_binary_predictions(
        evaluation_data
    )

    scenario_result = evaluate_scenarios(
        evaluation_data
    )

    aligned_hash = sha256(
        pd.util.hash_pandas_object(
            evaluation_data,
            index=True,
        ).to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()

    return SimpleNamespace(
        raw_splits=raw_splits,
        fit_result=fit_result,
        error_splits=error_splits,
        threshold_result=threshold_result,
        predictions=predictions,
        evaluation_data=evaluation_data,
        binary_result=binary_result,
        scenario_result=scenario_result,
        components_before=components_before,
        mean_before=mean_before,
        threshold_before=threshold_before,
        errors_before=errors_before,
        predictions_before=predictions_before,
        raw_test_before=raw_test_before,
        aligned_hash=aligned_hash,
    )


def test_phase_six_pipeline_contract(
    phase_six_evidence: SimpleNamespace,
) -> None:
    evidence = phase_six_evidence

    assert evidence.fit_result.n_components == 5

    assert (
        evidence.fit_result.achieved_explained_variance
        == pytest.approx(0.95811145295726)
    )

    assert (
        evidence.threshold_result.threshold
        == pytest.approx(0.19016111759041537)
    )

    assert (
        evidence.threshold_result.calibration_count
        == 800
    )

    assert evidence.evaluation_data.shape == (
        1800,
        4,
    )

    assert list(
        evidence.evaluation_data.columns
    ) == [
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    ]

    assert evidence.evaluation_data.index.equals(
        evidence.predictions.index
    )

    assert evidence.aligned_hash == (
        "6eeffc2ebc27964cedb037747d0438591c"
        "55518846aa906b46f5c2f1ea0705be"
    )


def test_phase_six_binary_metrics(
    phase_six_evidence: SimpleNamespace,
) -> None:
    result = phase_six_evidence.binary_result

    assert result.total == 1800
    assert result.normal_support == 800
    assert result.anomaly_support == 1000
    assert result.predicted_normal == 797
    assert result.predicted_anomaly == 1003

    assert result.true_negatives == 797
    assert result.false_positives == 3
    assert result.false_negatives == 0
    assert result.true_positives == 1000

    assert result.confusion_matrix == (
        (797, 3),
        (0, 1000),
    )

    assert result.precision == pytest.approx(
        0.9970089730807578
    )
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(
        0.9985022466300548
    )
    assert result.accuracy == pytest.approx(
        0.9983333333333333
    )
    assert (
        result.false_positive_rate
        == pytest.approx(0.00375)
    )
    assert (
        result.false_negative_rate
        == pytest.approx(0.0)
    )


def test_phase_six_scenario_results(
    phase_six_evidence: SimpleNamespace,
) -> None:
    result = (
        phase_six_evidence.scenario_result
        .set_index("scenario")
    )

    assert result.index.tolist() == [
        "normal",
        "brute_force",
        "dos",
        "exfiltration",
        "port_scan",
    ]

    assert result["observations"].to_dict() == {
        "normal": 800,
        "brute_force": 250,
        "dos": 250,
        "exfiltration": 250,
        "port_scan": 250,
    }

    assert (
        result["predicted_anomaly"].to_dict()
        == {
            "normal": 3,
            "brute_force": 250,
            "dos": 250,
            "exfiltration": 250,
            "port_scan": 250,
        }
    )

    assert result.loc[
        "normal",
        "predicted_anomaly_rate",
    ] == pytest.approx(0.00375)

    for scenario in (
        "brute_force",
        "dos",
        "exfiltration",
        "port_scan",
    ):
        assert result.loc[
            scenario,
            "predicted_anomaly_rate",
        ] == pytest.approx(1.0)

    means = result[
        "mean_reconstruction_error"
    ]

    assert (
        means["dos"]
        > means["port_scan"]
        > means["exfiltration"]
        > means["brute_force"]
        > means["normal"]
    )

    assert np.all(
        np.isfinite(
            result[
                [
                    "mean_reconstruction_error",
                    "median_reconstruction_error",
                    "maximum_reconstruction_error",
                ]
            ].to_numpy(dtype=np.float64)
        )
    )


def test_evaluation_preserves_frozen_state(
    phase_six_evidence: SimpleNamespace,
) -> None:
    evidence = phase_six_evidence

    np.testing.assert_array_equal(
        evidence.fit_result.model.components_,
        evidence.components_before,
    )

    np.testing.assert_array_equal(
        evidence.fit_result.model.mean_,
        evidence.mean_before,
    )

    assert (
        evidence.threshold_result.threshold
        == evidence.threshold_before
    )

    pd.testing.assert_series_equal(
        evidence.error_splits.test,
        evidence.errors_before,
    )

    pd.testing.assert_series_equal(
        evidence.predictions,
        evidence.predictions_before,
    )

    pd.testing.assert_frame_equal(
        evidence.raw_splits.test,
        evidence.raw_test_before,
    )
