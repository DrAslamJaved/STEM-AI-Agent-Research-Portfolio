from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from cyber_pca import (
    AnomalyThresholdResult,
    ManualPCA,
    PCAFitResult,
    ReconstructionErrorSplits,
    StandardizedDataSplits,
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    fit_normal_pca,
    generate_synthetic_network_data,
    predict_anomalies,
    split_normal_calibration_test,
    standardize_splits,
)


class BaselineDetectorRun(NamedTuple):
    standardized: StandardizedDataSplits
    fit_result: PCAFitResult
    errors: ReconstructionErrorSplits
    threshold: AnomalyThresholdResult
    predictions: pd.Series


def _run_baseline_detector() -> BaselineDetectorRun:
    dataset = generate_synthetic_network_data()

    raw_splits = split_normal_calibration_test(
        dataset,
        random_seed=42,
    )

    standardized = standardize_splits(raw_splits)

    fit_result = fit_normal_pca(
        standardized,
        explained_variance_target=0.95,
    )

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    threshold = calibrate_anomaly_threshold(
        errors,
        quantile=0.99,
        quantile_method="linear",
    )

    predictions = predict_anomalies(
        errors.test,
        threshold,
    )

    return BaselineDetectorRun(
        standardized=standardized,
        fit_result=fit_result,
        errors=errors,
        threshold=threshold,
        predictions=predictions,
    )


@pytest.fixture(scope="module")
def baseline_run() -> BaselineDetectorRun:
    return _run_baseline_detector()


def test_baseline_error_partition_contract(
    baseline_run: BaselineDetectorRun,
) -> None:
    expected_counts = {
        "normal_fit": 2400,
        "normal_calibration": 800,
        "test": 1800,
    }

    for split_name, expected_count in (
        expected_counts.items()
    ):
        standardized_frame = getattr(
            baseline_run.standardized,
            split_name,
        )
        error_series = getattr(
            baseline_run.errors,
            split_name,
        )

        assert len(error_series) == expected_count
        assert error_series.index.equals(
            standardized_frame.index
        )
        assert error_series.index.name == "flow_id"
        assert (
            error_series.name
            == "reconstruction_error"
        )
        assert error_series.dtype == np.dtype(
            np.float64
        )

        values = error_series.to_numpy(
            dtype=np.float64
        )

        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    assert baseline_run.fit_result.n_components == 5
    assert (
        baseline_run.fit_result
        .achieved_explained_variance
        >= 0.95
    )


def test_baseline_threshold_is_exact_calibration_quantile(
    baseline_run: BaselineDetectorRun,
) -> None:
    calibration_values = (
        baseline_run.errors.normal_calibration
        .to_numpy(dtype=np.float64)
    )

    expected_threshold = float(
        np.quantile(
            calibration_values,
            0.99,
            method="linear",
        )
    )

    assert (
        baseline_run.threshold.threshold
        == expected_threshold
    )
    assert baseline_run.threshold.quantile == 0.99
    assert (
        baseline_run.threshold.quantile_method
        == "linear"
    )
    assert (
        baseline_run.threshold.calibration_count
        == calibration_values.size
        == 800
    )

    above_threshold = np.count_nonzero(
        calibration_values > expected_threshold
    )

    assert 0 < above_threshold <= 8


def test_baseline_predictions_match_strict_rule(
    baseline_run: BaselineDetectorRun,
) -> None:
    expected = (
        baseline_run.errors.test
        > baseline_run.threshold.threshold
    ).astype(np.int8)

    expected.name = "is_anomaly"

    pd.testing.assert_series_equal(
        baseline_run.predictions,
        expected,
    )


def test_unlabelled_test_errors_show_synthetic_separation(
    baseline_run: BaselineDetectorRun,
) -> None:
    calibration_median = float(
        baseline_run.errors.normal_calibration
        .median()
    )
    test_median = float(
        baseline_run.errors.test.median()
    )

    assert (
        calibration_median
        < baseline_run.threshold.threshold
    )
    assert (
        test_median
        > baseline_run.threshold.threshold
    )
    assert test_median > calibration_median


def test_complete_baseline_detector_is_deterministic(
    baseline_run: BaselineDetectorRun,
) -> None:
    repeated = _run_baseline_detector()

    assert (
        repeated.fit_result.n_components
        == baseline_run.fit_result.n_components
    )
    assert (
        repeated.threshold
        == baseline_run.threshold
    )

    pd.testing.assert_series_equal(
        repeated.errors.normal_fit,
        baseline_run.errors.normal_fit,
    )
    pd.testing.assert_series_equal(
        repeated.errors.normal_calibration,
        baseline_run.errors.normal_calibration,
    )
    pd.testing.assert_series_equal(
        repeated.errors.test,
        baseline_run.errors.test,
    )
    pd.testing.assert_series_equal(
        repeated.predictions,
        baseline_run.predictions,
    )


def test_full_component_reconstruction_is_near_exact(
    baseline_run: BaselineDetectorRun,
) -> None:
    fitting_values = (
        baseline_run.standardized.normal_fit
        .to_numpy(dtype=np.float64)
    )

    full_model = ManualPCA(
        n_components=fitting_values.shape[1]
    )
    full_model.fit(fitting_values)

    frames = (
        baseline_run.standardized.normal_fit,
        baseline_run.standardized.normal_calibration,
        baseline_run.standardized.test,
    )

    for frame in frames:
        values = frame.to_numpy(dtype=np.float64)

        reconstructed = full_model.reconstruct(
            values
        )
        reconstruction_errors = (
            full_model.reconstruction_error(values)
        )

        np.testing.assert_allclose(
            reconstructed,
            values,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        assert (
            reconstruction_errors.max()
            < 1.0e-20
        )
