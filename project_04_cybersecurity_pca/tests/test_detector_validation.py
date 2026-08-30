from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cyber_pca.detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
    calibrate_anomaly_threshold,
    predict_anomalies,
)


def _error_series(
    values: list[float],
    *,
    prefix: str,
) -> pd.Series:
    return pd.Series(
        values,
        index=pd.Index(
            [
                f"{prefix}-{position}"
                for position in range(len(values))
            ],
            name="flow_id",
        ),
        name="reconstruction_error",
        dtype=np.float64,
    )


@pytest.fixture
def valid_errors() -> ReconstructionErrorSplits:
    return ReconstructionErrorSplits(
        normal_fit=_error_series(
            [0.10, 0.20, 0.30],
            prefix="fit",
        ),
        normal_calibration=_error_series(
            [0.15, 0.25, 0.35, 0.45],
            prefix="calibration",
        ),
        test=_error_series(
            [0.05, 0.80, 1.20],
            prefix="test",
        ),
    )


def test_calibration_rejects_wrong_errors_type() -> None:
    with pytest.raises(TypeError):
        calibrate_anomaly_threshold(object())


@pytest.mark.parametrize(
    "invalid_quantile",
    [
        0.0,
        1.0,
        -0.01,
        1.01,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_calibration_rejects_invalid_numeric_quantiles(
    valid_errors: ReconstructionErrorSplits,
    invalid_quantile: float,
) -> None:
    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            valid_errors,
            quantile=invalid_quantile,
        )


@pytest.mark.parametrize(
    "invalid_quantile",
    [
        True,
        False,
        "0.99",
        None,
    ],
)
def test_calibration_rejects_nonnumeric_quantiles(
    valid_errors: ReconstructionErrorSplits,
    invalid_quantile: object,
) -> None:
    with pytest.raises(TypeError):
        calibrate_anomaly_threshold(
            valid_errors,
            quantile=invalid_quantile,
        )


@pytest.mark.parametrize(
    "invalid_method",
    [
        None,
        1,
        True,
    ],
)
def test_calibration_rejects_nonstring_methods(
    valid_errors: ReconstructionErrorSplits,
    invalid_method: object,
) -> None:
    with pytest.raises(TypeError):
        calibrate_anomaly_threshold(
            valid_errors,
            quantile_method=invalid_method,
        )


@pytest.mark.parametrize(
    "invalid_method",
    [
        "nearest",
        "lower",
        "LINEAR",
    ],
)
def test_calibration_rejects_unsupported_methods(
    valid_errors: ReconstructionErrorSplits,
    invalid_method: str,
) -> None:
    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            valid_errors,
            quantile_method=invalid_method,
        )


def test_calibration_rejects_empty_series(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    empty = pd.Series(
        [],
        index=pd.Index([], name="flow_id"),
        name="reconstruction_error",
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=empty,
            )
        )


def test_calibration_rejects_wrong_series_name(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    invalid = valid_errors.normal_calibration.rename(
        "error"
    )

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


def test_calibration_rejects_wrong_index_name(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    invalid = valid_errors.normal_calibration.copy()
    invalid.index = invalid.index.rename("identifier")

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


def test_calibration_rejects_missing_flow_ids(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    invalid = valid_errors.normal_calibration.copy()
    invalid.index = pd.Index(
        [
            "calibration-0",
            None,
            "calibration-2",
            "calibration-3",
        ],
        name="flow_id",
    )

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


def test_calibration_rejects_duplicate_flow_ids(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    invalid = valid_errors.normal_calibration.copy()
    invalid.index = pd.Index(
        [
            "calibration-0",
            "calibration-0",
            "calibration-2",
            "calibration-3",
        ],
        name="flow_id",
    )

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_calibration_rejects_nonfinite_errors(
    valid_errors: ReconstructionErrorSplits,
    invalid_value: float,
) -> None:
    invalid = valid_errors.normal_calibration.copy()
    invalid.iloc[1] = invalid_value

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


def test_calibration_rejects_negative_errors(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    invalid = valid_errors.normal_calibration.copy()
    invalid.iloc[1] = -0.01

    with pytest.raises(ValueError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


def test_calibration_rejects_nonnumeric_errors(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    invalid = pd.Series(
        ["small", "medium", "large", "extreme"],
        index=valid_errors.normal_calibration.index,
        name="reconstruction_error",
        dtype=object,
    )

    with pytest.raises(TypeError):
        calibrate_anomaly_threshold(
            replace(
                valid_errors,
                normal_calibration=invalid,
            )
        )


def test_calibration_does_not_mutate_errors(
    valid_errors: ReconstructionErrorSplits,
) -> None:
    original = (
        valid_errors.normal_calibration.copy(
            deep=True
        )
    )

    calibrate_anomaly_threshold(valid_errors)

    pd.testing.assert_series_equal(
        valid_errors.normal_calibration,
        original,
    )

@pytest.fixture
def valid_prediction_errors() -> pd.Series:
    return _error_series(
        [0.10, 0.50, 0.75, 1.20],
        prefix="prediction",
    )


@pytest.fixture
def valid_threshold_result() -> AnomalyThresholdResult:
    return AnomalyThresholdResult(
        threshold=0.50,
        quantile=0.99,
        quantile_method="linear",
        calibration_count=800,
    )


@pytest.mark.parametrize(
    "invalid_errors",
    [
        object(),
        [0.1, 0.2],
        np.array([0.1, 0.2]),
    ],
)
def test_prediction_rejects_wrong_errors_type(
    valid_threshold_result: AnomalyThresholdResult,
    invalid_errors: object,
) -> None:
    with pytest.raises(TypeError):
        predict_anomalies(
            invalid_errors,
            valid_threshold_result,
        )


@pytest.mark.parametrize(
    "invalid_threshold_result",
    [
        object(),
        0.50,
        None,
    ],
)
def test_prediction_rejects_wrong_threshold_result_type(
    valid_prediction_errors: pd.Series,
    invalid_threshold_result: object,
) -> None:
    with pytest.raises(TypeError):
        predict_anomalies(
            valid_prediction_errors,
            invalid_threshold_result,
        )


def test_prediction_rejects_empty_errors(
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    empty = pd.Series(
        [],
        index=pd.Index([], name="flow_id"),
        name="reconstruction_error",
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        predict_anomalies(
            empty,
            valid_threshold_result,
        )


def test_prediction_rejects_wrong_series_name(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    invalid = valid_prediction_errors.rename("error")

    with pytest.raises(ValueError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


def test_prediction_rejects_wrong_index_name(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    invalid = valid_prediction_errors.copy()
    invalid.index = invalid.index.rename("identifier")

    with pytest.raises(ValueError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


def test_prediction_rejects_missing_flow_ids(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    invalid = valid_prediction_errors.copy()
    invalid.index = pd.Index(
        [
            "prediction-0",
            None,
            "prediction-2",
            "prediction-3",
        ],
        name="flow_id",
    )

    with pytest.raises(ValueError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


def test_prediction_rejects_duplicate_flow_ids(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    invalid = valid_prediction_errors.copy()
    invalid.index = pd.Index(
        [
            "prediction-0",
            "prediction-0",
            "prediction-2",
            "prediction-3",
        ],
        name="flow_id",
    )

    with pytest.raises(ValueError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


def test_prediction_rejects_nonnumeric_errors(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    invalid = pd.Series(
        ["small", "medium", "large", "extreme"],
        index=valid_prediction_errors.index,
        name="reconstruction_error",
        dtype=object,
    )

    with pytest.raises(TypeError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_prediction_rejects_nonfinite_errors(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
    invalid_value: float,
) -> None:
    invalid = valid_prediction_errors.copy()
    invalid.iloc[1] = invalid_value

    with pytest.raises(ValueError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


def test_prediction_rejects_negative_errors(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    invalid = valid_prediction_errors.copy()
    invalid.iloc[1] = -0.01

    with pytest.raises(ValueError):
        predict_anomalies(
            invalid,
            valid_threshold_result,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        True,
        "0.50",
        None,
    ],
)
def test_prediction_rejects_nonreal_thresholds(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
    invalid_threshold: object,
) -> None:
    invalid_result = replace(
        valid_threshold_result,
        threshold=invalid_threshold,
    )

    with pytest.raises(TypeError):
        predict_anomalies(
            valid_prediction_errors,
            invalid_result,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        np.nan,
        np.inf,
        -np.inf,
        -0.01,
    ],
)
def test_prediction_rejects_invalid_numeric_thresholds(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
    invalid_threshold: float,
) -> None:
    invalid_result = replace(
        valid_threshold_result,
        threshold=invalid_threshold,
    )

    with pytest.raises(ValueError):
        predict_anomalies(
            valid_prediction_errors,
            invalid_result,
        )


def test_prediction_does_not_mutate_errors(
    valid_prediction_errors: pd.Series,
    valid_threshold_result: AnomalyThresholdResult,
) -> None:
    original = valid_prediction_errors.copy(
        deep=True
    )

    predict_anomalies(
        valid_prediction_errors,
        valid_threshold_result,
    )

    pd.testing.assert_series_equal(
        valid_prediction_errors,
        original,
    )
