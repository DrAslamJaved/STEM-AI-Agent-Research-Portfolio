"""Normal-only PCA fitting and component scoring."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from cyber_pca.pca_manual import ManualPCA
from cyber_pca.preprocessing import (
    StandardizedDataSplits,
)
from cyber_pca.synthetic_data import FEATURE_COLUMNS


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PCAFitResult:
    """Fitted retained PCA model and full variance evidence."""

    model: ManualPCA
    n_components: int
    explained_variance_target: float
    achieved_explained_variance: float
    full_explained_variance: FloatArray
    full_explained_variance_ratio: FloatArray
    full_cumulative_explained_variance: FloatArray


@dataclass(frozen=True)
class PCAScoreSplits:
    """Principal-component scores aligned by flow ID."""

    normal_fit: pd.DataFrame
    normal_calibration: pd.DataFrame
    test: pd.DataFrame


def _validate_variance_target(
    value: object,
) -> float:
    """Validate the explained-variance target."""

    if isinstance(value, bool) or not isinstance(
        value,
        Real,
    ):
        raise TypeError(
            "explained_variance_target must be a "
            "real number."
        )

    float_value = float(value)

    if not np.isfinite(float_value):
        raise ValueError(
            "explained_variance_target must be finite."
        )

    if not 0.0 < float_value <= 1.0:
        raise ValueError(
            "explained_variance_target must satisfy "
            "0 < target <= 1."
        )

    return float_value


def _readonly_float64_copy(
    values: ArrayLike,
) -> FloatArray:
    """Return an immutable float64 copy."""

    copied = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    copied.setflags(write=False)

    return copied


def select_n_components(
    explained_variance_ratios: ArrayLike,
    *,
    explained_variance_target: float = 0.95,
) -> int:
    """Select the minimum cumulative-variance count."""

    target = _validate_variance_target(
        explained_variance_target
    )

    ratios = np.asarray(
        explained_variance_ratios,
        dtype=np.float64,
    )

    if ratios.ndim != 1:
        raise ValueError(
            "explained_variance_ratios must be "
            "one-dimensional."
        )

    if ratios.size == 0:
        raise ValueError(
            "explained_variance_ratios must not be "
            "empty."
        )

    if not np.all(np.isfinite(ratios)):
        raise ValueError(
            "explained_variance_ratios must be finite."
        )

    if np.any(ratios < 0.0):
        raise ValueError(
            "explained_variance_ratios must be "
            "nonnegative."
        )

    total = float(ratios.sum())

    if total <= 0.0:
        raise ValueError(
            "explained_variance_ratios must have a "
            "positive sum."
        )

    if not np.isclose(
        total,
        1.0,
        rtol=1.0e-9,
        atol=1.0e-12,
    ):
        raise ValueError(
            "explained_variance_ratios must sum to 1."
        )

    normalized_ratios = ratios / total

    cumulative = np.cumsum(
        normalized_ratios,
        dtype=np.float64,
    )

    cumulative[-1] = 1.0

    selected_index = int(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
    )

    return selected_index + 1


def _validate_standardized_splits(
    splits: object,
) -> StandardizedDataSplits:
    """Validate standardized PCA input partitions."""

    if not isinstance(
        splits,
        StandardizedDataSplits,
    ):
        raise TypeError(
            "splits must be a "
            "StandardizedDataSplits instance."
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

        if tuple(frame.columns) != FEATURE_COLUMNS:
            raise ValueError(
                f"{name} columns must exactly match "
                "FEATURE_COLUMNS."
            )

        if frame.index.name != "flow_id":
            raise ValueError(
                f"{name} index must be named flow_id."
            )

        if frame.index.hasnans:
            raise ValueError(
                f"{name} contains missing flow IDs."
            )

        if frame.index.duplicated().any():
            raise ValueError(
                f"{name} contains duplicate flow IDs."
            )

        values = frame.to_numpy(
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} contains nonfinite values."
            )

        identifier_sets[name] = set(frame.index)

    if (
        splits.normal_fit.shape[0] < 2
    ):
        raise ValueError(
            "normal_fit must contain at least two "
            "observations."
        )

    fit_ids = identifier_sets["normal_fit"]
    calibration_ids = identifier_sets[
        "normal_calibration"
    ]
    test_ids = identifier_sets["test"]

    if not fit_ids.isdisjoint(calibration_ids):
        raise ValueError(
            "Fitting and calibration flow IDs overlap."
        )

    if not fit_ids.isdisjoint(test_ids):
        raise ValueError(
            "Fitting and test flow IDs overlap."
        )

    if not calibration_ids.isdisjoint(test_ids):
        raise ValueError(
            "Calibration and test flow IDs overlap."
        )

    return splits


def fit_normal_pca(
    splits: StandardizedDataSplits,
    *,
    explained_variance_target: float = 0.95,
    eigenvalue_tolerance: float = 1.0e-12,
) -> PCAFitResult:
    """Fit PCA using standardized normal fitting data."""

    validated_splits = (
        _validate_standardized_splits(splits)
    )

    target = _validate_variance_target(
        explained_variance_target
    )

    fit_values = (
        validated_splits.normal_fit.to_numpy(
            dtype=np.float64,
        )
    )

    full_model = ManualPCA(
        n_components=None,
        eigenvalue_tolerance=(
            eigenvalue_tolerance
        ),
    )

    full_model.fit(fit_values)

    full_ratios = (
        full_model.all_explained_variance_ratio_
    )

    selected_count = select_n_components(
        full_ratios,
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
        selected_model.all_explained_variance_,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise RuntimeError(
            "Full and selected PCA fits produced "
            "inconsistent eigenvalues."
        )

    full_explained_variance = (
        _readonly_float64_copy(
            full_model.all_explained_variance_
        )
    )

    full_explained_variance_ratio = (
        _readonly_float64_copy(
            full_model.all_explained_variance_ratio_
        )
    )

    full_cumulative = np.cumsum(
        full_explained_variance_ratio,
        dtype=np.float64,
    )

    full_cumulative[-1] = 1.0
    full_cumulative.setflags(write=False)

    achieved = float(
        full_cumulative[selected_count - 1]
    )

    return PCAFitResult(
        model=selected_model,
        n_components=selected_count,
        explained_variance_target=target,
        achieved_explained_variance=achieved,
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


def _score_frame(
    model: ManualPCA,
    standardized_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Transform one aligned standardized partition."""

    values = standardized_frame.to_numpy(
        dtype=np.float64,
    )

    scores = model.transform(values)

    component_columns = [
        f"PC{index}"
        for index in range(
            1,
            model.n_components_ + 1,
        )
    ]

    return pd.DataFrame(
        scores,
        columns=component_columns,
        index=standardized_frame.index.copy(),
        dtype=np.float64,
    )


def transform_pca_splits(
    splits: StandardizedDataSplits,
    fit_result: PCAFitResult,
) -> PCAScoreSplits:
    """Transform every partition with one PCA model."""

    validated_splits = (
        _validate_standardized_splits(splits)
    )

    if not isinstance(fit_result, PCAFitResult):
        raise TypeError(
            "fit_result must be a PCAFitResult."
        )

    model = fit_result.model

    if not hasattr(model, "components_"):
        raise ValueError(
            "fit_result contains an unfitted PCA model."
        )

    if model.n_features_in_ != len(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "PCA model feature count does not match "
            "FEATURE_COLUMNS."
        )

    return PCAScoreSplits(
        normal_fit=_score_frame(
            model,
            validated_splits.normal_fit,
        ),
        normal_calibration=_score_frame(
            model,
            validated_splits.normal_calibration,
        ),
        test=_score_frame(
            model,
            validated_splits.test,
        ),
    )
