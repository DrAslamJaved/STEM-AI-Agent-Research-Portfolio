"""Leakage-safe feature engineering for time-series forecasting."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from time_series_agent.exceptions import FeatureEngineeringError


@dataclass
class FeatureSet:
    """Features and aligned target ready for model training."""

    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    input_rows: int
    dropped_rows: int


@dataclass(frozen=True)
class FeatureSummary:
    """Machine-readable feature-engineering summary."""

    input_rows: int
    output_rows: int
    dropped_rows: int
    feature_count: int
    feature_names: tuple[str, ...]
    first_usable_timestamp: str
    last_usable_timestamp: str
    maximum_lag: int
    maximum_rolling_window: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable summary."""
        return asdict(self)


def _validate_periods(
    periods: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    """Validate positive, unique lag or window lengths."""
    if not periods:
        raise FeatureEngineeringError(
            f"At least one {label} is required."
        )

    for period in periods:
        if (
            not isinstance(period, int)
            or isinstance(period, bool)
            or period <= 0
        ):
            raise FeatureEngineeringError(
                f"Every {label} must be a positive integer."
            )

    if len(set(periods)) != len(periods):
        raise FeatureEngineeringError(
            f"Duplicate {label} values are not allowed."
        )

    return tuple(sorted(periods))


def build_lag_feature_set(
    data: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    lags: tuple[int, ...] = (1, 24, 168),
    rolling_windows: tuple[int, ...] = (24, 168),
) -> FeatureSet:
    """Build lag, rolling, calendar, and trend features.

    All target-derived features for time ``t`` are calculated from
    observations no later than ``t - 1``.
    """
    required_columns = {
        timestamp_column,
        target_column,
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise FeatureEngineeringError(
            f"Feature engineering requires: {missing_text}"
        )

    validated_lags = _validate_periods(
        lags,
        "lag",
    )
    validated_windows = _validate_periods(
        rolling_windows,
        "rolling window",
    )

    timestamps = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
    )
    target = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise FeatureEngineeringError(
            "Feature engineering found invalid timestamps."
        )

    if target.isna().any():
        raise FeatureEngineeringError(
            "Feature engineering found invalid targets."
        )

    if not np.isfinite(target.to_numpy()).all():
        raise FeatureEngineeringError(
            "Feature engineering found nonfinite targets."
        )

    timestamp_index = pd.DatetimeIndex(timestamps)

    if timestamp_index.has_duplicates:
        raise FeatureEngineeringError(
            "Feature timestamps must be unique."
        )

    if not timestamp_index.is_monotonic_increasing:
        raise FeatureEngineeringError(
            "Feature timestamps must be chronologically ordered."
        )

    indexed_target = pd.Series(
        target.to_numpy(dtype="float64"),
        index=timestamp_index,
        name="target",
    )

    feature_data: dict[str, Any] = {}

    for lag in validated_lags:
        feature_data[f"lag_{lag}"] = indexed_target.shift(lag)

    past_target = indexed_target.shift(1)

    for window in validated_windows:
        feature_data[f"rolling_mean_{window}"] = (
            past_target.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )
        feature_data[f"rolling_std_{window}"] = (
            past_target.rolling(
                window=window,
                min_periods=window,
            ).std(ddof=0)
        )

    hour_angle = (
        2 * np.pi * timestamp_index.hour / 24
    )
    weekday_angle = (
        2 * np.pi * timestamp_index.dayofweek / 7
    )

    feature_data["hour_sin"] = np.sin(hour_angle)
    feature_data["hour_cos"] = np.cos(hour_angle)
    feature_data["day_of_week_sin"] = np.sin(
        weekday_angle
    )
    feature_data["day_of_week_cos"] = np.cos(
        weekday_angle
    )
    feature_data["trend_index"] = np.arange(
        len(indexed_target),
        dtype="float64",
    )

    features = pd.DataFrame(
        feature_data,
        index=timestamp_index,
    )

    valid_rows = features.notna().all(axis=1)

    usable_features = features.loc[
        valid_rows
    ].copy()
    usable_target = indexed_target.loc[
        valid_rows
    ].copy()

    if usable_features.empty:
        raise FeatureEngineeringError(
            "No usable rows remain after feature construction."
        )

    if not usable_features.index.equals(
        usable_target.index
    ):
        raise FeatureEngineeringError(
            "Features and target are not temporally aligned."
        )

    return FeatureSet(
        features=usable_features,
        target=usable_target,
        feature_names=tuple(usable_features.columns),
        input_rows=len(data),
        dropped_rows=int((~valid_rows).sum()),
    )

def build_recursive_feature_row(
    history: pd.Series,
    forecast_timestamp: pd.Timestamp,
    trend_index: int,
    lags: tuple[int, ...] = (1, 24, 168),
    rolling_windows: tuple[int, ...] = (24, 168),
) -> pd.DataFrame:
    """Build one future feature row from past history only."""
    validated_lags = _validate_periods(
        lags,
        "lag",
    )
    validated_windows = _validate_periods(
        rolling_windows,
        "rolling window",
    )

    if not isinstance(history, pd.Series):
        raise FeatureEngineeringError(
            "Recursive history must be a pandas Series."
        )

    if not isinstance(history.index, pd.DatetimeIndex):
        raise FeatureEngineeringError(
            "Recursive history must use a DatetimeIndex."
        )

    if history.empty:
        raise FeatureEngineeringError(
            "Recursive history must not be empty."
        )

    if history.index.has_duplicates:
        raise FeatureEngineeringError(
            "Recursive history timestamps must be unique."
        )

    if not history.index.is_monotonic_increasing:
        raise FeatureEngineeringError(
            "Recursive history must be chronologically ordered."
        )

    numeric_history = pd.to_numeric(
        history,
        errors="coerce",
    )

    if (
        numeric_history.isna().any()
        or not np.isfinite(
            numeric_history.to_numpy()
        ).all()
    ):
        raise FeatureEngineeringError(
            "Recursive history contains invalid values."
        )

    timestamp = pd.Timestamp(forecast_timestamp)

    if timestamp <= history.index[-1]:
        raise FeatureEngineeringError(
            "Forecast timestamp must occur after history."
        )

    if (
        not isinstance(trend_index, int)
        or isinstance(trend_index, bool)
        or trend_index < 0
    ):
        raise FeatureEngineeringError(
            "Trend index must be a nonnegative integer."
        )

    required_history = max(
        max(validated_lags),
        max(validated_windows),
    )

    if len(numeric_history) < required_history:
        raise FeatureEngineeringError(
            f"Recursive forecasting requires at least "
            f"{required_history} historical observations."
        )

    feature_data: dict[str, float] = {}

    for lag in validated_lags:
        feature_data[f"lag_{lag}"] = float(
            numeric_history.iloc[-lag]
        )

    for window in validated_windows:
        window_values = numeric_history.iloc[-window:]

        feature_data[f"rolling_mean_{window}"] = float(
            window_values.mean()
        )
        feature_data[f"rolling_std_{window}"] = float(
            window_values.std(ddof=0)
        )

    hour_angle = (
        2 * np.pi * timestamp.hour / 24
    )
    weekday_angle = (
        2 * np.pi * timestamp.dayofweek / 7
    )

    feature_data["hour_sin"] = float(
        np.sin(hour_angle)
    )
    feature_data["hour_cos"] = float(
        np.cos(hour_angle)
    )
    feature_data["day_of_week_sin"] = float(
        np.sin(weekday_angle)
    )
    feature_data["day_of_week_cos"] = float(
        np.cos(weekday_angle)
    )
    feature_data["trend_index"] = float(trend_index)

    return pd.DataFrame(
        [feature_data],
        index=pd.DatetimeIndex([timestamp]),
    )

def create_feature_summary(
    feature_set: FeatureSet,
    lags: tuple[int, ...] = (1, 24, 168),
    rolling_windows: tuple[int, ...] = (24, 168),
) -> FeatureSummary:
    """Summarize one constructed feature set."""
    return FeatureSummary(
        input_rows=feature_set.input_rows,
        output_rows=len(feature_set.features),
        dropped_rows=feature_set.dropped_rows,
        feature_count=len(feature_set.feature_names),
        feature_names=feature_set.feature_names,
        first_usable_timestamp=str(
            feature_set.features.index.min()
        ),
        last_usable_timestamp=str(
            feature_set.features.index.max()
        ),
        maximum_lag=max(lags),
        maximum_rolling_window=max(
            rolling_windows
        ),
    )


def save_feature_summary(
    summary: FeatureSummary,
    output_path: str | Path,
) -> None:
    """Save the feature summary as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )