"""Tests for expanding-window time-series validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.evaluation import (
    create_expanding_window_plots,
    evaluate_models_expanding_window,
    expanding_window_splits,
    save_expanding_window_results,
    summarize_expanding_window_results,
)
from time_series_agent.exceptions import EvaluationError


def make_series(
    observations: int = 240,
) -> pd.Series:
    """Create deterministic hourly data with daily seasonality."""
    time_index = np.arange(observations)

    values = (
        100
        + 15 * np.sin(2 * np.pi * time_index / 24)
        + 0.05 * time_index
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


def model_factories() -> dict[str, object]:
    """Create baseline model factories."""
    return {
        "mean": lambda: MeanForecaster(),
        "naive": lambda: NaiveForecaster(),
        "seasonal_24": (
            lambda: SeasonalNaiveForecaster(24)
        ),
    }


def test_expanding_splits_grow_without_leakage() -> None:
    """Training should expand while every test remains future data."""
    series = make_series(observations=100)

    splits = expanding_window_splits(
        series=series,
        initial_train_size=50,
        horizon=10,
        step=10,
    )

    assert len(splits) == 5

    for expected_fold, (
        fold,
        training,
        test,
    ) in enumerate(splits, start=1):
        assert fold == expected_fold
        assert len(training) == 50 + 10 * (fold - 1)
        assert len(test) == 10
        assert training.index.max() < test.index.min()


@pytest.mark.parametrize(
    ("initial_train_size", "horizon", "step"),
    [
        (0, 10, 10),
        (50, 0, 10),
        (50, 10, 0),
        (95, 10, 10),
    ],
)
def test_invalid_split_parameters_are_rejected(
    initial_train_size: int,
    horizon: int,
    step: int,
) -> None:
    """Invalid split sizes should produce clear errors."""
    with pytest.raises(EvaluationError):
        expanding_window_splits(
            series=make_series(observations=100),
            initial_train_size=initial_train_size,
            horizon=horizon,
            step=step,
        )


def test_expanding_evaluation_returns_every_fold_model_pair() -> None:
    """Every model should be evaluated on every temporal fold."""
    results = evaluate_models_expanding_window(
        series=make_series(),
        model_factories=model_factories(),
        initial_train_size=120,
        horizon=24,
        step=24,
        mase_period=24,
    )

    assert results["fold"].nunique() == 5
    assert results["model"].nunique() == 3
    assert len(results) == 15
    assert results[
        ["mae", "rmse", "smape", "mase"]
    ].isna().sum().sum() == 0


def test_expanding_summary_ranks_models() -> None:
    """Aggregate results should contain variability and ranks."""
    results = evaluate_models_expanding_window(
        series=make_series(),
        model_factories=model_factories(),
        initial_train_size=120,
        horizon=24,
        step=24,
        mase_period=24,
    )

    summary = summarize_expanding_window_results(results)

    assert len(summary) == 3
    assert set(summary["folds"]) == {5}
    assert summary["mae_rank"].min() == 1
    assert summary["mae_fold_wins"].sum() == 5
    assert summary["mean_mae"].is_monotonic_increasing


def test_cross_validation_artifacts_can_be_saved(
    tmp_path: Path,
) -> None:
    """Fold results, summaries, and plots should be saved."""
    results = evaluate_models_expanding_window(
        series=make_series(),
        model_factories=model_factories(),
        initial_train_size=120,
        horizon=24,
        step=24,
        mase_period=24,
    )
    summary = summarize_expanding_window_results(results)

    fold_path = tmp_path / "folds.csv"
    summary_path = tmp_path / "summary.csv"

    save_expanding_window_results(
        fold_results=results,
        summary=summary,
        fold_results_path=fold_path,
        summary_path=summary_path,
    )

    plot_paths = create_expanding_window_plots(
        fold_results=results,
        summary=summary,
        output_directory=tmp_path,
    )

    assert fold_path.is_file()
    assert summary_path.is_file()
    assert all(path.is_file() for path in plot_paths)
    assert all(path.stat().st_size > 0 for path in plot_paths)