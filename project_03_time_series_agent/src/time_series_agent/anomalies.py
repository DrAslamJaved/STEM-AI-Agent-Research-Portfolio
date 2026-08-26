"""Robust residual-based anomaly detection."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from time_series_agent.exceptions import (
    AnomalyDetectionError,
)


MODIFIED_Z_CONSTANT = 0.6744897501960817

REQUIRED_INPUT_COLUMNS = {
    "timestamp",
    "residual",
    "is_known_closure",
}

REQUIRED_OUTPUT_COLUMNS = {
    "modified_z_score",
    "absolute_modified_z_score",
    "is_statistical_anomaly",
    "is_actionable_anomaly",
    "anomaly_type",
    "anomaly_severity",
}


@dataclass(frozen=True)
class AnomalyDetectionSummary:
    """Aggregate evidence from residual anomaly detection."""

    row_count: int
    reference_row_count: int
    known_closure_count: int
    residual_center: float
    residual_mad: float
    modified_z_threshold: float
    statistical_anomaly_count: int
    actionable_anomaly_count: int
    positive_anomaly_count: int
    negative_anomaly_count: int
    closure_statistical_anomaly_count: int
    normal_nonclosure_count: int
    actionable_anomaly_rate_percent: float
    maximum_absolute_modified_z_score: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-serializable summary."""
        return {
            "method": "median_mad_modified_z_score",
            "residual_definition": "actual_minus_forecast",
            "known_closures_used_for_calibration": "no",
            "row_count": self.row_count,
            "reference_row_count": (
                self.reference_row_count
            ),
            "known_closure_count": (
                self.known_closure_count
            ),
            "residual_center": self.residual_center,
            "residual_mad": self.residual_mad,
            "modified_z_threshold": (
                self.modified_z_threshold
            ),
            "statistical_anomaly_count": (
                self.statistical_anomaly_count
            ),
            "actionable_anomaly_count": (
                self.actionable_anomaly_count
            ),
            "positive_anomaly_count": (
                self.positive_anomaly_count
            ),
            "negative_anomaly_count": (
                self.negative_anomaly_count
            ),
            "closure_statistical_anomaly_count": (
                self.closure_statistical_anomaly_count
            ),
            "normal_nonclosure_count": (
                self.normal_nonclosure_count
            ),
            "actionable_anomaly_rate_percent": (
                self.actionable_anomaly_rate_percent
            ),
            "maximum_absolute_modified_z_score": (
                self.maximum_absolute_modified_z_score
            ),
        }


def _validate_threshold(
    threshold: float,
) -> float:
    """Validate the modified-z threshold."""
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise AnomalyDetectionError(
            "'threshold' must be a positive finite number."
        )

    return float(threshold)


def _validate_minimum_reference_rows(
    minimum_reference_rows: int,
) -> int:
    """Validate the minimum calibration-sample size."""
    if (
        not isinstance(minimum_reference_rows, int)
        or isinstance(minimum_reference_rows, bool)
        or minimum_reference_rows <= 0
    ):
        raise AnomalyDetectionError(
            "'minimum_reference_rows' must be a "
            "positive integer."
        )

    return minimum_reference_rows


def _validate_residual_frame(
    residuals: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and copy residual evidence."""
    if not isinstance(residuals, pd.DataFrame):
        raise AnomalyDetectionError(
            "'residuals' must be a pandas DataFrame."
        )

    if residuals.empty:
        raise AnomalyDetectionError(
            "The residual table cannot be empty."
        )

    missing_columns = sorted(
        REQUIRED_INPUT_COLUMNS
        - set(residuals.columns)
    )

    if missing_columns:
        raise AnomalyDetectionError(
            "Residual table is missing required columns: "
            + ", ".join(missing_columns)
        )

    validated = residuals.copy(deep=True)

    timestamps = pd.to_datetime(
        validated["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise AnomalyDetectionError(
            "Residual table contains invalid timestamps."
        )

    if timestamps.duplicated().any():
        raise AnomalyDetectionError(
            "Residual timestamps must be unique."
        )

    if not timestamps.is_monotonic_increasing:
        raise AnomalyDetectionError(
            "Residual timestamps must be chronological."
        )

    numeric_residuals = pd.to_numeric(
        validated["residual"],
        errors="coerce",
    )

    if numeric_residuals.isna().any():
        raise AnomalyDetectionError(
            "Residual values must be numeric and complete."
        )

    if not np.isfinite(
        numeric_residuals.to_numpy(dtype="float64")
    ).all():
        raise AnomalyDetectionError(
            "Residual values must be finite."
        )

    closure_flags = validated[
        "is_known_closure"
    ]

    if not pd.api.types.is_bool_dtype(
        closure_flags.dtype
    ):
        raise AnomalyDetectionError(
            "'is_known_closure' must contain Boolean values."
        )

    validated["timestamp"] = timestamps
    validated["residual"] = (
        numeric_residuals.astype("float64")
    )
    validated["is_known_closure"] = (
        closure_flags.astype(bool)
    )

    return validated


def detect_residual_anomalies(
    residuals: pd.DataFrame,
    threshold: float = 3.5,
    minimum_reference_rows: int = 20,
) -> tuple[pd.DataFrame, AnomalyDetectionSummary]:
    """Detect robust anomalies while excluding known closures."""
    validated_threshold = _validate_threshold(
        threshold
    )
    validated_minimum = (
        _validate_minimum_reference_rows(
            minimum_reference_rows
        )
    )
    labeled = _validate_residual_frame(residuals)

    reference_mask = ~labeled[
        "is_known_closure"
    ]
    reference_residuals = labeled.loc[
        reference_mask,
        "residual",
    ]

    if len(reference_residuals) < validated_minimum:
        raise AnomalyDetectionError(
            "Too few nonclosure residuals are available "
            "for anomaly calibration."
        )

    residual_center = float(
        reference_residuals.median()
    )

    residual_mad = float(
        np.median(
            np.abs(
                reference_residuals.to_numpy(
                    dtype="float64"
                )
                - residual_center
            )
        )
    )

    if (
        not np.isfinite(residual_mad)
        or residual_mad <= 0
    ):
        raise AnomalyDetectionError(
            "Residual MAD must be positive for robust "
            "anomaly detection."
        )

    modified_scores = (
        MODIFIED_Z_CONSTANT
        * (
            labeled["residual"]
            - residual_center
        )
        / residual_mad
    )

    absolute_scores = modified_scores.abs()

    statistical_anomalies = (
        absolute_scores >= validated_threshold
    )

    actionable_anomalies = (
        statistical_anomalies
        & ~labeled["is_known_closure"]
    )

    positive_anomalies = (
        actionable_anomalies
        & modified_scores.gt(0)
    )

    negative_anomalies = (
        actionable_anomalies
        & modified_scores.lt(0)
    )

    labeled["modified_z_score"] = (
        modified_scores.astype("float64")
    )
    labeled["absolute_modified_z_score"] = (
        absolute_scores.astype("float64")
    )
    labeled["is_statistical_anomaly"] = (
        statistical_anomalies.astype(bool)
    )
    labeled["is_actionable_anomaly"] = (
        actionable_anomalies.astype(bool)
    )

    labeled["anomaly_type"] = np.select(
        [
            labeled["is_known_closure"],
            positive_anomalies,
            negative_anomalies,
        ],
        [
            "known_closure",
            "positive_demand_spike",
            "negative_demand_drop",
        ],
        default="normal",
    )

    labeled["anomaly_severity"] = np.select(
        [
            labeled["is_known_closure"],
            actionable_anomalies
            & absolute_scores.ge(7.5),
            actionable_anomalies
            & absolute_scores.ge(5.0),
            actionable_anomalies,
        ],
        [
            "known_closure",
            "extreme",
            "high",
            "moderate",
        ],
        default="normal",
    )

    closure_statistical_count = int(
        (
            labeled["is_known_closure"]
            & statistical_anomalies
        ).sum()
    )

    actionable_count = int(
        actionable_anomalies.sum()
    )

    reference_count = int(
        reference_mask.sum()
    )

    summary = AnomalyDetectionSummary(
        row_count=int(len(labeled)),
        reference_row_count=reference_count,
        known_closure_count=int(
            labeled["is_known_closure"].sum()
        ),
        residual_center=residual_center,
        residual_mad=residual_mad,
        modified_z_threshold=(
            validated_threshold
        ),
        statistical_anomaly_count=int(
            statistical_anomalies.sum()
        ),
        actionable_anomaly_count=(
            actionable_count
        ),
        positive_anomaly_count=int(
            positive_anomalies.sum()
        ),
        negative_anomaly_count=int(
            negative_anomalies.sum()
        ),
        closure_statistical_anomaly_count=(
            closure_statistical_count
        ),
        normal_nonclosure_count=(
            reference_count - actionable_count
        ),
        actionable_anomaly_rate_percent=float(
            100 * actionable_count / reference_count
        ),
        maximum_absolute_modified_z_score=float(
            absolute_scores.max()
        ),
    )

    return labeled, summary

def select_top_anomaly_candidates(
    labeled_anomalies: pd.DataFrame,
    limit: int = 10,
) -> pd.DataFrame:
    """Select actionable candidates by absolute score."""
    if not isinstance(
        labeled_anomalies,
        pd.DataFrame,
    ):
        raise AnomalyDetectionError(
            "'labeled_anomalies' must be a DataFrame."
        )

    required_columns = {
        "is_actionable_anomaly",
        "absolute_modified_z_score",
    }

    missing_columns = sorted(
        required_columns
        - set(labeled_anomalies.columns)
    )

    if missing_columns:
        raise AnomalyDetectionError(
            "Anomaly table is missing selection columns: "
            + ", ".join(missing_columns)
        )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        raise AnomalyDetectionError(
            "'limit' must be a positive integer."
        )

    selected = (
        labeled_anomalies.loc[
            labeled_anomalies[
                "is_actionable_anomaly"
            ]
        ]
        .sort_values(
            "absolute_modified_z_score",
            ascending=False,
        )
        .head(limit)
        .copy()
    )

    return selected.reset_index(drop=True)

def save_anomaly_results(
    labeled_anomalies: pd.DataFrame,
    summary: AnomalyDetectionSummary,
    labels_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Save anomaly labels and their structured summary."""
    if not isinstance(labeled_anomalies, pd.DataFrame):
        raise AnomalyDetectionError(
            "'labeled_anomalies' must be a DataFrame."
        )

    missing_columns = sorted(
        REQUIRED_OUTPUT_COLUMNS
        - set(labeled_anomalies.columns)
    )

    if missing_columns:
        raise AnomalyDetectionError(
            "Anomaly table is missing output columns: "
            + ", ".join(missing_columns)
        )

    if not isinstance(
        summary,
        AnomalyDetectionSummary,
    ):
        raise AnomalyDetectionError(
            "'summary' must be an "
            "AnomalyDetectionSummary."
        )

    labels_destination = Path(labels_path)
    summary_destination = Path(summary_path)

    labels_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labeled_anomalies.to_csv(
        labels_destination,
        index=False,
    )

    summary_destination.write_text(
        json.dumps(
            summary.to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return (
        labels_destination,
        summary_destination,
    )