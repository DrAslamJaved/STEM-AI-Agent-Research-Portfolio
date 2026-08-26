"""Integration tests for machine-learning temporal evaluation."""

import numpy as np
import pandas as pd

from time_series_agent.baselines import (
    SeasonalNaiveForecaster,
)
from time_series_agent.evaluation import (
    evaluate_models_expanding_window,
    summarize_expanding_window_results,
)
from time_series_agent.machine_learning import (
    RecursiveGradientBoostingForecaster,
)


def make_series(
    observations: int = 600,
) -> pd.Series:
    """Create deterministic hourly seasonal data."""
    time_index = np.arange(observations)

    values = (
        200
        + 0.03 * time_index
        + 40 * np.sin(2 * np.pi * time_index / 24)
        + 15 * np.sin(2 * np.pi * time_index / 168)
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


def test_gradient_boosting_runs_across_expanding_windows() -> None:
    """Gradient Boosting should work with temporal evaluation."""
    factories = {
        "seasonal_naive_168": (
            lambda: SeasonalNaiveForecaster(
                seasonal_period=168,
                frequency="h",
            )
        ),
        "gradient_boosting_recursive": (
            lambda: RecursiveGradientBoostingForecaster(
                lags=(1, 24, 168),
                rolling_windows=(24, 168),
                n_estimators=30,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=3,
                random_state=42,
                clip_nonnegative=True,
                frequency="h",
            )
        ),
    }

    results = evaluate_models_expanding_window(
        series=make_series(),
        model_factories=factories,
        initial_train_size=504,
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

    assert set(results["model"]) == {
        "seasonal_naive_168",
        "gradient_boosting_recursive",
    }

    assert results[
        ["mae", "rmse", "smape", "mase"]
    ].isna().sum().sum() == 0

    assert (
        "raw_negative_forecast_count"
        in results.columns
    )
    assert results[
        "raw_negative_forecast_count"
    ].ge(0).all()

    assert (
        "total_raw_negative_forecasts"
        in summary.columns
    )

    candidate_results = results.loc[
        results["model"].eq(
            "gradient_boosting_recursive"
        )
    ]

    assert len(candidate_results) == 2
    assert candidate_results[
        "test_rows"
    ].eq(48).all()