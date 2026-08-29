"""PCA reconstruction-error anomaly detection."""

from __future__ import annotations
from numbers import Real
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cyber_pca.pca_workflow import (
    PCAFitResult,
    StandardizedDataSplits,
    transform_pca_splits,
)


@dataclass(frozen=True)
class ReconstructionErrorSplits:
    """Reconstruction errors aligned with data partitions."""

    normal_fit: pd.Series
    normal_calibration: pd.Series
    test: pd.Series


@dataclass(frozen=True)
class AnomalyThresholdResult:
    """Normal-calibration anomaly threshold evidence."""

    threshold: float
    quantile: float
    quantile_method: str
    calibration_count: int


def compute_reconstruction_errors(
    splits: StandardizedDataSplits,
    fit_result: PCAFitResult,
) -> ReconstructionErrorSplits:
    """Calculate PCA reconstruction errors for each partition."""
    transform_pca_splits(
        splits,
        fit_result,
    )

    def calculate_partition_errors(
        feature_frame: pd.DataFrame,
    ) -> pd.Series:
        matrix = feature_frame.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        error_values = np.asarray(
            fit_result.model.reconstruction_error(
                matrix
            ),
            dtype=np.float64,
        )

        expected_shape = (
            feature_frame.shape[0],
        )

        if error_values.shape != expected_shape:
            raise RuntimeError(
                "Reconstruction errors have an "
                "unexpected shape."
            )

        if not np.all(np.isfinite(error_values)):
            raise RuntimeError(
                "Reconstruction errors must be finite."
            )

        if np.any(error_values < 0.0):
            raise RuntimeError(
                "Reconstruction errors must be "
                "nonnegative."
            )

        return pd.Series(
            error_values,
            index=feature_frame.index.copy(),
            name="reconstruction_error",
            dtype=np.float64,
            copy=True,
        )

    return ReconstructionErrorSplits(
        normal_fit=calculate_partition_errors(
            splits.normal_fit
        ),
        normal_calibration=(
            calculate_partition_errors(
                splits.normal_calibration
            )
        ),
        test=calculate_partition_errors(
            splits.test
        ),
    )

def calibrate_anomaly_threshold(
    errors: ReconstructionErrorSplits,
    *,
    quantile: float = 0.99,
    quantile_method: str = "linear",
) -> AnomalyThresholdResult:
    """Calibrate a threshold from normal-calibration errors."""
    if not isinstance(
        errors,
        ReconstructionErrorSplits,
    ):
        raise TypeError(
            "errors must be a "
            "ReconstructionErrorSplits instance."
        )

    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, Real)
    ):
        raise TypeError(
            "quantile must be a real number."
        )

    validated_quantile = float(quantile)

    if (
        not np.isfinite(validated_quantile)
        or not 0.0 < validated_quantile < 1.0
    ):
        raise ValueError(
            "quantile must be finite and satisfy "
            "0 < quantile < 1."
        )

    if not isinstance(quantile_method, str):
        raise TypeError(
            "quantile_method must be a string."
        )

    if quantile_method != "linear":
        raise ValueError(
            "quantile_method must be 'linear'."
        )

    calibration_errors = (
        errors.normal_calibration
    )

    if not isinstance(
        calibration_errors,
        pd.Series,
    ):
        raise TypeError(
            "normal_calibration errors must be "
            "a pandas Series."
        )

    if calibration_errors.empty:
        raise ValueError(
            "normal_calibration errors must not "
            "be empty."
        )

    if (
        calibration_errors.name
        != "reconstruction_error"
    ):
        raise ValueError(
            "normal_calibration error series must "
            "be named reconstruction_error."
        )

    if calibration_errors.index.name != "flow_id":
        raise ValueError(
            "normal_calibration error index must "
            "be named flow_id."
        )

    if calibration_errors.index.hasnans:
        raise ValueError(
            "normal_calibration errors contain "
            "missing flow IDs."
        )

    if calibration_errors.index.duplicated().any():
        raise ValueError(
            "normal_calibration errors contain "
            "duplicate flow IDs."
        )
    try:
        calibration_values = (
            calibration_errors.to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "normal calibration errors must contain "
            "numeric values."
        ) from error

    if not np.all(
        np.isfinite(calibration_values)
    ):
        raise ValueError(
            "normal_calibration errors must be "
            "finite."
        )

    if np.any(calibration_values < 0.0):
        raise ValueError(
            "normal_calibration errors must be "
            "nonnegative."
        )

    threshold = float(
        np.quantile(
            calibration_values,
            validated_quantile,
            method=quantile_method,
        )
    )

    if not np.isfinite(threshold):
        raise RuntimeError(
            "Calculated anomaly threshold is "
            "not finite."
        )

    if threshold < 0.0:
        raise RuntimeError(
            "Calculated anomaly threshold is "
            "negative."
        )

    return AnomalyThresholdResult(
        threshold=threshold,
        quantile=validated_quantile,
        quantile_method=quantile_method,
        calibration_count=(
            calibration_values.shape[0]
        ),
    )

def predict_anomalies(
    errors: pd.Series,
    threshold_result: AnomalyThresholdResult,
) -> pd.Series:
    """Predict anomalies using a calibrated reconstruction-error threshold."""

    if not isinstance(errors, pd.Series):
        raise TypeError(
            "errors must be a pandas Series."
        )

    if not isinstance(
        threshold_result,
        AnomalyThresholdResult,
    ):
        raise TypeError(
            "threshold_result must be an "
            "AnomalyThresholdResult."
        )

    if errors.empty:
        raise ValueError(
            "errors must not be empty."
        )

    if errors.name != "reconstruction_error":
        raise ValueError(
            "errors must be named "
            "'reconstruction_error'."
        )

    if errors.index.name != "flow_id":
        raise ValueError(
            "errors index must be named 'flow_id'."
        )

    if errors.index.hasnans:
        raise ValueError(
            "errors contains missing flow IDs."
        )

    if errors.index.duplicated().any():
        raise ValueError(
            "errors contains duplicate flow IDs."
        )

    try:
        values = errors.to_numpy(
            dtype=np.float64,
            copy=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "errors must contain numeric values."
        ) from error

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "errors contains nonfinite values."
        )

    if np.any(values < 0.0):
        raise ValueError(
            "reconstruction errors must be "
            "nonnegative."
        )

    threshold = threshold_result.threshold

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
    ):
        raise TypeError(
            "threshold must be a real number."
        )

    threshold_value = float(threshold)

    if not np.isfinite(threshold_value):
        raise ValueError(
            "threshold must be finite."
        )

    if threshold_value < 0.0:
        raise ValueError(
            "threshold must be nonnegative."
        )

    predictions = (
        values > threshold_value
    ).astype(
        np.int8,
        copy=False,
    )

    return pd.Series(
        predictions,
        index=errors.index.copy(),
        name="is_anomaly",
        dtype=np.int8,
    )
