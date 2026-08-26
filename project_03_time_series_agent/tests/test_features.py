"""Tests for leakage-safe time-series feature engineering."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from time_series_agent.exceptions import FeatureEngineeringError
from time_series_agent.features import (
    build_lag_feature_set,
    create_feature_summary,
    save_feature_summary,
)


def make_data(
    observations: int = 240,
) -> pd.DataFrame:
    """Create a simple ordered hourly target."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=observations,
                freq="h",
            ),
            "target": np.arange(
                observations,
                dtype="float64",
            ),
        }
    )


def test_feature_shape_and_names() -> None:
    """Maximum lag should determine the first usable row."""
    feature_set = build_lag_feature_set(
        make_data(),
        timestamp_column="timestamp",
        target_column="target",
    )

    assert len(feature_set.features) == 72
    assert feature_set.dropped_rows == 168
    assert len(feature_set.feature_names) == 12
    assert feature_set.features.index[0] == pd.Timestamp(
        "2026-01-08 00:00:00"
    )


def test_lags_reference_only_earlier_targets() -> None:
    """Lag values should match their documented past positions."""
    feature_set = build_lag_feature_set(
        make_data(),
        timestamp_column="timestamp",
        target_column="target",
    )

    first_row = feature_set.features.iloc[0]

    assert first_row["lag_1"] == 167
    assert first_row["lag_24"] == 144
    assert first_row["lag_168"] == 0


def test_rolling_features_end_at_previous_hour() -> None:
    """Rolling statistics must exclude the current target."""
    feature_set = build_lag_feature_set(
        make_data(),
        timestamp_column="timestamp",
        target_column="target",
    )

    first_row = feature_set.features.iloc[0]

    assert first_row["rolling_mean_24"] == pytest.approx(
        np.mean(np.arange(144, 168))
    )
    assert first_row["rolling_mean_168"] == pytest.approx(
        np.mean(np.arange(0, 168))
    )


def test_current_target_change_does_not_change_current_features() -> None:
    """Changing y_t must not change features used to predict y_t."""
    original = make_data()
    changed = original.copy(deep=True)

    changed.loc[200, "target"] += 10_000

    original_set = build_lag_feature_set(
        original,
        timestamp_column="timestamp",
        target_column="target",
    )
    changed_set = build_lag_feature_set(
        changed,
        timestamp_column="timestamp",
        target_column="target",
    )

    timestamp_200 = original.loc[200, "timestamp"]
    timestamp_201 = original.loc[201, "timestamp"]

    pd.testing.assert_series_equal(
        original_set.features.loc[timestamp_200],
        changed_set.features.loc[timestamp_200],
    )

    assert (
        original_set.target.loc[timestamp_200]
        != changed_set.target.loc[timestamp_200]
    )
    assert (
        original_set.features.loc[
            timestamp_201,
            "lag_1",
        ]
        != changed_set.features.loc[
            timestamp_201,
            "lag_1",
        ]
    )


def test_cyclical_calendar_features_are_bounded() -> None:
    """Sine and cosine features should remain within [-1, 1]."""
    feature_set = build_lag_feature_set(
        make_data(),
        timestamp_column="timestamp",
        target_column="target",
    )

    calendar_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    assert (
        feature_set.features[
            calendar_columns
        ].abs().le(1).all().all()
    )


@pytest.mark.parametrize(
    ("lags", "windows"),
    [
        ((0,), (24,)),
        ((1,), (-24,)),
    ],
)
def test_invalid_feature_periods_are_rejected(
    lags: tuple[int, ...],
    windows: tuple[int, ...],
) -> None:
    """Lag and rolling periods must be positive."""
    with pytest.raises(
        FeatureEngineeringError,
        match="positive integer",
    ):
        build_lag_feature_set(
            make_data(),
            timestamp_column="timestamp",
            target_column="target",
            lags=lags,
            rolling_windows=windows,
        )


def test_unsorted_timestamps_are_rejected() -> None:
    """Feature engineering must not silently sort bad input."""
    data = make_data().iloc[::-1].reset_index(drop=True)

    with pytest.raises(
        FeatureEngineeringError,
        match="chronologically ordered",
    ):
        build_lag_feature_set(
            data,
            timestamp_column="timestamp",
            target_column="target",
        )


def test_feature_summary_can_be_saved(
    tmp_path: Path,
) -> None:
    """Feature metadata should be saved as valid JSON."""
    feature_set = build_lag_feature_set(
        make_data(),
        timestamp_column="timestamp",
        target_column="target",
    )
    summary = create_feature_summary(feature_set)
    output_path = tmp_path / "features.json"

    save_feature_summary(
        summary,
        output_path,
    )

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved["input_rows"] == 240
    assert saved["output_rows"] == 72
    assert saved["dropped_rows"] == 168
    assert saved["feature_count"] == 12