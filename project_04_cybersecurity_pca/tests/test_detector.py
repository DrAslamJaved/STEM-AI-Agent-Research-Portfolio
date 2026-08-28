"""Tests for PCA reconstruction-error anomaly detection."""

from dataclasses import fields, is_dataclass, replace

import numpy as np
import pandas as pd
import pytest

from cyber_pca.detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
    calibrate_anomaly_threshold,
    compute_reconstruction_errors,
    predict_anomalies,
)
from cyber_pca.pca_workflow import (
    PCAFitResult,
    fit_normal_pca,
)
from cyber_pca.preprocessing import (
    StandardizedDataSplits,
    split_normal_calibration_test,
    standardize_splits,
)
from cyber_pca.synthetic_data import (
    generate_synthetic_network_data,
)


@pytest.fixture(scope="module")
def fitted_detector_inputs() -> tuple[
    StandardizedDataSplits,
    PCAFitResult,
]:
    """Build one deterministic fitted PCA workflow."""
    dataset = generate_synthetic_network_data(
        n_normal=300,
        n_attack_per_type=50,
        random_seed=42,
    )

    standardized = standardize_splits(
        split_normal_calibration_test(
            dataset,
            random_seed=42,
        )
    )

    fit_result = fit_normal_pca(
        standardized,
        explained_variance_target=0.95,
    )

    return standardized, fit_result


def test_detector_result_dataclass_contracts() -> None:
    assert is_dataclass(ReconstructionErrorSplits)
    assert is_dataclass(AnomalyThresholdResult)

    assert [
        field.name
        for field in fields(ReconstructionErrorSplits)
    ] == [
        "normal_fit",
        "normal_calibration",
        "test",
    ]

    assert [
        field.name
        for field in fields(AnomalyThresholdResult)
    ] == [
        "threshold",
        "quantile",
        "quantile_method",
        "calibration_count",
    ]


def test_detector_functions_are_callable() -> None:
    assert callable(compute_reconstruction_errors)
    assert callable(calibrate_anomaly_threshold)
    assert callable(predict_anomalies)


def test_reconstruction_error_series_contract(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    partition_pairs = (
        (
            standardized.normal_fit,
            errors.normal_fit,
        ),
        (
            standardized.normal_calibration,
            errors.normal_calibration,
        ),
        (
            standardized.test,
            errors.test,
        ),
    )

    for feature_frame, error_series in partition_pairs:
        assert isinstance(error_series, pd.Series)
        assert error_series.name == "reconstruction_error"
        assert error_series.index.name == "flow_id"
        assert error_series.dtype == np.dtype(np.float64)

        assert len(error_series) == len(feature_frame)

        pd.testing.assert_index_equal(
            error_series.index,
            feature_frame.index,
        )

        values = error_series.to_numpy(
            dtype=np.float64
        )

        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)

    assert float(errors.normal_fit.mean()) > 0.0


def test_reconstruction_errors_match_manual_formula(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    matrix = (
        standardized
        .normal_calibration
        .to_numpy(dtype=np.float64)
    )

    reconstructed = fit_result.model.reconstruct(
        matrix
    )

    expected = np.mean(
        (matrix - reconstructed) ** 2,
        axis=1,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        errors.normal_calibration.to_numpy(),
        expected,
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_reconstruction_errors_agree_with_manual_pca_method(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    expected = fit_result.model.reconstruction_error(
        standardized.test.to_numpy(
            dtype=np.float64
        )
    )

    np.testing.assert_allclose(
        errors.test.to_numpy(),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_reconstruction_error_execution_is_deterministic(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    first = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    second = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    pd.testing.assert_series_equal(
        first.normal_fit,
        second.normal_fit,
        check_exact=True,
    )

    pd.testing.assert_series_equal(
        first.normal_calibration,
        second.normal_calibration,
        check_exact=True,
    )

    pd.testing.assert_series_equal(
        first.test,
        second.test,
        check_exact=True,
    )

def test_threshold_matches_explicit_numpy_quantile(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    result = calibrate_anomaly_threshold(
        errors,
        quantile=0.99,
        quantile_method="linear",
    )

    expected = float(
        np.quantile(
            errors.normal_calibration.to_numpy(
                dtype=np.float64
            ),
            0.99,
            method="linear",
        )
    )

    assert result.threshold == expected
    assert result.quantile == 0.99
    assert result.quantile_method == "linear"
    assert result.calibration_count == len(
        errors.normal_calibration
    )
    assert np.isfinite(result.threshold)
    assert result.threshold >= 0.0


def test_threshold_uses_only_normal_calibration_errors(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    baseline = calibrate_anomaly_threshold(
        errors,
        quantile=0.99,
        quantile_method="linear",
    )

    modified_errors = replace(
        errors,
        normal_fit=(
            errors.normal_fit * 1.0e12
        ),
        test=(
            errors.test + 1.0e12
        ),
    )

    modified = calibrate_anomaly_threshold(
        modified_errors,
        quantile=0.99,
        quantile_method="linear",
    )

    assert modified == baseline


def test_threshold_calibration_is_deterministic(
    fitted_detector_inputs: tuple[
        StandardizedDataSplits,
        PCAFitResult,
    ],
) -> None:
    standardized, fit_result = fitted_detector_inputs

    errors = compute_reconstruction_errors(
        standardized,
        fit_result,
    )

    first = calibrate_anomaly_threshold(
        errors,
        quantile=0.99,
        quantile_method="linear",
    )

    second = calibrate_anomaly_threshold(
        errors,
        quantile=0.99,
        quantile_method="linear",
    )

    assert first == second

def test_predictions_use_strict_greater_than_rule() -> None:
    errors = pd.Series(
        [0.49, 0.50, 0.50000001],
        index=pd.Index(
            ["flow-a", "flow-b", "flow-c"],
            name="flow_id",
        ),
        name="reconstruction_error",
        dtype=np.float64,
    )

    threshold_result = AnomalyThresholdResult(
        threshold=0.50,
        quantile=0.99,
        quantile_method="linear",
        calibration_count=100,
    )

    predictions = predict_anomalies(
        errors,
        threshold_result,
    )

    expected = pd.Series(
        [0, 0, 1],
        index=errors.index,
        name="is_anomaly",
        dtype=np.int8,
    )

    pd.testing.assert_series_equal(
        predictions,
        expected,
    )


def test_prediction_series_contract() -> None:
    errors = pd.Series(
        [1.2, 0.1, 0.8],
        index=pd.Index(
            ["flow-30", "flow-10", "flow-20"],
            name="flow_id",
        ),
        name="reconstruction_error",
        dtype=np.float64,
    )

    threshold_result = AnomalyThresholdResult(
        threshold=0.75,
        quantile=0.99,
        quantile_method="linear",
        calibration_count=800,
    )

    predictions = predict_anomalies(
        errors,
        threshold_result,
    )

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "is_anomaly"
    assert predictions.index.equals(errors.index)
    assert predictions.index.name == "flow_id"
    assert predictions.dtype == np.dtype(np.int8)
    assert set(predictions.unique()) <= {0, 1}


def test_prediction_execution_is_deterministic() -> None:
    errors = pd.Series(
        [0.2, 0.9, 0.4, 1.1],
        index=pd.Index(
            ["flow-1", "flow-2", "flow-3", "flow-4"],
            name="flow_id",
        ),
        name="reconstruction_error",
        dtype=np.float64,
    )
    original_errors = errors.copy(deep=True)

    threshold_result = AnomalyThresholdResult(
        threshold=0.5,
        quantile=0.99,
        quantile_method="linear",
        calibration_count=800,
    )

    first = predict_anomalies(
        errors,
        threshold_result,
    )
    second = predict_anomalies(
        errors,
        threshold_result,
    )

    pd.testing.assert_series_equal(first, second)
    pd.testing.assert_series_equal(
        errors,
        original_errors,
    )
