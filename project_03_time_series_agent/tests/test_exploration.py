"""Tests for reproducible exploratory time-series analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from time_series_agent.exceptions import ExplorationError
from time_series_agent.exploration import (
    compute_exploration_summary,
    create_exploration_figures,
    save_exploration_summary,
)


def make_exploration_data(
    observations: int = 240,
) -> pd.DataFrame:
    """Create deterministic hourly data with daily seasonality."""
    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=observations,
        freq="h",
    )

    time_index = np.arange(observations)
    random_generator = np.random.default_rng(42)

    target = (
        100
        + 0.05 * time_index
        + 20 * np.sin(2 * np.pi * time_index / 24)
        + random_generator.normal(0, 2, observations)
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "target": target,
        }
    )


def test_summary_contains_valid_statistics() -> None:
    """The numerical summary should describe the supplied series."""
    data = make_exploration_data()

    summary = compute_exploration_summary(
        data,
        timestamp_column="timestamp",
        target_column="target",
    )

    assert summary.row_count == 240
    assert summary.target_minimum < summary.target_maximum
    assert summary.target_standard_deviation > 0
    assert -1 <= summary.autocorrelation_lag_24 <= 1
    assert -1 <= summary.autocorrelation_lag_168 <= 1
    assert 0 <= summary.adf_p_value <= 1


def test_exploration_rejects_missing_columns() -> None:
    """The exploration module should require named columns."""
    data = pd.DataFrame({"target": range(60)})

    with pytest.raises(
        ExplorationError,
        match="timestamp",
    ):
        compute_exploration_summary(
            data,
            timestamp_column="timestamp",
            target_column="target",
        )


def test_exploration_rejects_short_series() -> None:
    """Very short series should not receive unstable diagnostics."""
    data = make_exploration_data(observations=24)

    with pytest.raises(
        ExplorationError,
        match="at least 48 observations",
    ):
        compute_exploration_summary(
            data,
            timestamp_column="timestamp",
            target_column="target",
        )


def test_exploration_does_not_modify_input() -> None:
    """Exploratory analysis should leave supplied data unchanged."""
    data = make_exploration_data()
    original = data.copy(deep=True)

    compute_exploration_summary(
        data,
        timestamp_column="timestamp",
        target_column="target",
    )

    pd.testing.assert_frame_equal(data, original)


def test_figures_and_summary_are_saved(
    tmp_path: Path,
) -> None:
    """Exploration should create five figures and one JSON file."""
    data = make_exploration_data()

    figure_directory = tmp_path / "figures"
    summary_path = tmp_path / "summary.json"

    summary = compute_exploration_summary(
        data,
        timestamp_column="timestamp",
        target_column="target",
    )

    figure_paths = create_exploration_figures(
        data=data,
        timestamp_column="timestamp",
        target_column="target",
        output_directory=figure_directory,
        seasonal_period=24,
        maximum_lag=48,
    )

    save_exploration_summary(summary, summary_path)

    assert len(figure_paths) == 5
    assert all(path.is_file() for path in figure_paths)
    assert all(path.stat().st_size > 0 for path in figure_paths)

    saved_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert saved_summary["row_count"] == 240