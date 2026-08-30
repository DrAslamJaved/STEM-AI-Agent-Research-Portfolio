"""Label-blind PCA detection for official UNSW-NB15 data."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from cyber_pca.detector import (
    AnomalyThresholdResult,
    ReconstructionErrorSplits,
    calibrate_anomaly_threshold,
    predict_anomalies,
)
from cyber_pca.pca_manual import ManualPCA
from cyber_pca.pca_workflow import (
    PCAFitResult,
    select_n_components,
)
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWStandardizedDataSplits,
)


@dataclass(frozen=True)
class UNSWDetectionResult:
    """Frozen UNSW-NB15 detector outputs before label access."""

    fit_result: PCAFitResult
    reconstruction_errors: ReconstructionErrorSplits
    threshold_result: AnomalyThresholdResult
    test_predictions: pd.Series


def _readonly_float64_copy(
    values: object,
) -> np.ndarray:
    """Return an immutable float64 array copy."""

    result = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )

    result.setflags(write=False)

    return result


def _validate_variance_target(
    value: object,
) -> float:
    """Validate the cumulative explained-variance target."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TypeError(
            "explained_variance_target must be "
            "numeric."
        )

    target = float(value)

    if not np.isfinite(target):
        raise ValueError(
            "explained_variance_target must be "
            "finite."
        )

    if not 0.0 < target <= 1.0:
        raise ValueError(
            "explained_variance_target must be "
            "in the interval (0, 1]."
        )

    return target


def _validate_unsw_standardized_splits(
    splits: object,
) -> UNSWStandardizedDataSplits:
    """Validate the schema-independent UNSW model matrices."""

    if not isinstance(
        splits,
        UNSWStandardizedDataSplits,
    ):
        raise TypeError(
            "splits must be a "
            "UNSWStandardizedDataSplits instance."
        )

    if not isinstance(
        splits.preprocessor,
        UNSWPreprocessor,
    ):
        raise TypeError(
            "splits.preprocessor must be a "
            "UNSWPreprocessor instance."
        )

    scaler = splits.preprocessor.scaler

    if not isinstance(
        scaler,
        StandardScaler,
    ):
        raise TypeError(
            "splits.preprocessor.scaler must be "
            "a StandardScaler instance."
        )

    feature_names = tuple(
        splits.preprocessor.feature_names
    )

    if not feature_names:
        raise ValueError(
            "feature_names must not be empty."
        )

    if not all(
        isinstance(name, str) and name
        for name in feature_names
    ):
        raise TypeError(
            "feature_names must contain nonempty "
            "strings."
        )

    if len(set(feature_names)) != len(
        feature_names
    ):
        raise ValueError(
            "feature_names must be unique."
        )

    named_frames = (
        ("normal_fit", splits.normal_fit),
        (
            "normal_calibration",
            splits.normal_calibration,
        ),
        ("test", splits.test),
    )

    identifier_sets: dict[
        str,
        set[object],
    ] = {}

    for name, frame in named_frames:
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas "
                "DataFrame."
            )

        if frame.empty:
            raise ValueError(
                f"{name} must not be empty."
            )

        if tuple(frame.columns) != feature_names:
            raise ValueError(
                f"{name} columns must exactly "
                "match feature_names."
            )

        if frame.index.name != "flow_id":
            raise ValueError(
                f"{name} index must be named "
                "flow_id."
            )

        if frame.index.hasnans:
            raise ValueError(
                f"{name} contains missing "
                "flow IDs."
            )

        if frame.index.duplicated().any():
            raise ValueError(
                f"{name} contains duplicate "
                "flow IDs."
            )

        for column in frame.columns:
            series = frame[column]

            if (
                pd.api.types.is_bool_dtype(
                    series.dtype
                )
                or not pd.api.types.is_numeric_dtype(
                    series.dtype
                )
            ):
                raise TypeError(
                    f"{name} model features must "
                    "be numeric and nonboolean."
                )

        values = frame.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} contains nonfinite "
                "model values."
            )

        identifier_sets[name] = set(
            frame.index.tolist()
        )

    if splits.normal_fit.shape[0] < 2:
        raise ValueError(
            "normal_fit must contain at least "
            "two observations."
        )

    fit_ids = identifier_sets[
        "normal_fit"
    ]
    calibration_ids = identifier_sets[
        "normal_calibration"
    ]
    test_ids = identifier_sets["test"]

    if not fit_ids.isdisjoint(
        calibration_ids
    ):
        raise ValueError(
            "Fitting and calibration flow IDs "
            "overlap."
        )

    if not fit_ids.isdisjoint(test_ids):
        raise ValueError(
            "Fitting and test flow IDs overlap."
        )

    if not calibration_ids.isdisjoint(
        test_ids
    ):
        raise ValueError(
            "Calibration and test flow IDs "
            "overlap."
        )

    try:
        check_is_fitted(scaler)
    except NotFittedError as exception:
        raise ValueError(
            "The UNSW StandardScaler must be "
            "fitted."
        ) from exception

    if (
        int(scaler.n_features_in_)
        != len(feature_names)
    ):
        raise ValueError(
            "The fitted scaler feature count "
            "does not match feature_names."
        )

    return splits


def fit_unsw_normal_pca(
    splits: UNSWStandardizedDataSplits,
    *,
    explained_variance_target: float = 0.95,
    eigenvalue_tolerance: float = 1e-12,
) -> PCAFitResult:
    """Fit PCA using UNSW normal fitting observations only."""

    validated_splits = (
        _validate_unsw_standardized_splits(
            splits
        )
    )

    target = _validate_variance_target(
        explained_variance_target
    )

    fit_values = (
        validated_splits.normal_fit.to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )

    full_model = ManualPCA(
        n_components=None,
        eigenvalue_tolerance=(
            eigenvalue_tolerance
        ),
    )

    full_model.fit(fit_values)

    selected_count = select_n_components(
        full_model
        .all_explained_variance_ratio_,
        explained_variance_target=target,
    )

    selected_model = ManualPCA(
        n_components=selected_count,
        eigenvalue_tolerance=(
            eigenvalue_tolerance
        ),
    )

    selected_model.fit(fit_values)

    if not np.allclose(
        full_model.all_explained_variance_,
        selected_model
        .all_explained_variance_,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise RuntimeError(
            "Full and selected PCA fits produced "
            "inconsistent eigenvalues."
        )

    full_explained_variance = (
        _readonly_float64_copy(
            full_model
            .all_explained_variance_
        )
    )

    full_explained_variance_ratio = (
        _readonly_float64_copy(
            full_model
            .all_explained_variance_ratio_
        )
    )

    full_cumulative = np.cumsum(
        full_explained_variance_ratio,
        dtype=np.float64,
    )

    full_cumulative[-1] = 1.0
    full_cumulative.setflags(
        write=False
    )

    achieved = float(
        full_cumulative[
            selected_count - 1
        ]
    )

    return PCAFitResult(
        model=selected_model,
        n_components=selected_count,
        explained_variance_target=target,
        achieved_explained_variance=(
            achieved
        ),
        full_explained_variance=(
            full_explained_variance
        ),
        full_explained_variance_ratio=(
            full_explained_variance_ratio
        ),
        full_cumulative_explained_variance=(
            full_cumulative
        ),
    )


def compute_unsw_reconstruction_errors(
    splits: UNSWStandardizedDataSplits,
    fit_result: PCAFitResult,
) -> ReconstructionErrorSplits:
    """Calculate reconstruction errors for each UNSW split."""

    validated_splits = (
        _validate_unsw_standardized_splits(
            splits
        )
    )

    if not isinstance(
        fit_result,
        PCAFitResult,
    ):
        raise TypeError(
            "fit_result must be a "
            "PCAFitResult instance."
        )

    if not isinstance(
        fit_result.model,
        ManualPCA,
    ):
        raise TypeError(
            "fit_result.model must be a "
            "ManualPCA instance."
        )

    def calculate_partition_errors(
        feature_frame: pd.DataFrame,
    ) -> pd.Series:
        matrix = feature_frame.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        error_values = np.asarray(
            fit_result.model
            .reconstruction_error(matrix),
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

        if not np.all(
            np.isfinite(error_values)
        ):
            raise RuntimeError(
                "Reconstruction errors must be "
                "finite."
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
        normal_fit=(
            calculate_partition_errors(
                validated_splits.normal_fit
            )
        ),
        normal_calibration=(
            calculate_partition_errors(
                validated_splits
                .normal_calibration
            )
        ),
        test=calculate_partition_errors(
            validated_splits.test
        ),
    )


def run_unsw_detection(
    splits: UNSWStandardizedDataSplits,
    *,
    explained_variance_target: float = 0.95,
    eigenvalue_tolerance: float = 1e-12,
    threshold_quantile: float = 0.99,
    quantile_method: str = "linear",
) -> UNSWDetectionResult:
    """Fit, calibrate, and predict without accessing labels."""

    fit_result = fit_unsw_normal_pca(
        splits,
        explained_variance_target=(
            explained_variance_target
        ),
        eigenvalue_tolerance=(
            eigenvalue_tolerance
        ),
    )

    reconstruction_errors = (
        compute_unsw_reconstruction_errors(
            splits,
            fit_result,
        )
    )

    threshold_result = (
        calibrate_anomaly_threshold(
            reconstruction_errors,
            quantile=threshold_quantile,
            quantile_method=quantile_method,
        )
    )

    test_predictions = (
        predict_anomalies(
            reconstruction_errors.test,
            threshold_result,
        ).rename(
            "predicted_anomaly"
        )
    )

    return UNSWDetectionResult(
        fit_result=fit_result,
        reconstruction_errors=(
            reconstruction_errors
        ),
        threshold_result=threshold_result,
        test_predictions=test_predictions,
    )
