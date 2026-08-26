"""Integration tests for Holt-Winters temporal evaluation."""

import numpy as np
import pandas as pd

from time_series_agent.baselines import (
    SeasonalNaiveForecaster,
)
from time_series_agent.classical import (
    HoltWintersForecaster,
)
from time_series_agent.evaluation import (
    evaluate_models_expanding_window,
    summarize_expanding_window_results,
)


def make_series(
    observations: int = 336,
) -> pd.Series:
    """Create deterministic hourly daily-seasonal data."""
    time_index = np.arange(observations)

    values = (
        120
        + 0.04 * time_index
        + 25 * np.sin(2 * np.pi * time_index / 24)
    )

    return pd.Series(
        values,
        index=pd.date_range(
            "2026-01-01 00:00:00",
            periods=observations,
            freq="h",
        ),
        dtype="float64",
        name="target",
    )


def test_holt_winters_runs_across_expanding_windows() -> None:
    """Holt-Winters should integrate with temporal evaluation."""
    factories = {
        "seasonal_naive_24": (
            lambda: SeasonalNaiveForecaster(24)
        ),
        "holt_winters_24": (
            lambda: HoltWintersForecaster(
                seasonal_period=24,
                clip_nonnegative=True,
            )
        ),
    }

    results = evaluate_models_expanding_window(
        series=make_series(),
        model_factories=factories,
        initial_train_size=240,
        horizon=48,
        step=48,
        mase_period=24,
    )

    summary = summarize_expanding_window_results(
        results
    )

    assert results["fold"].nunique() == 2
    assert results["model"].nunique() == 2
    assert len(results) == 4
    assert "raw_negative_forecast_count" in results.columns
    assert (
        results[
            "raw_negative_forecast_count"
        ].ge(0).all()
    )
    assert (
        "total_raw_negative_forecasts"
        in summary.columns
    )
    assert results[
        ["mae", "rmse", "smape", "mase"]
    ].isna().sum().sum() == 0