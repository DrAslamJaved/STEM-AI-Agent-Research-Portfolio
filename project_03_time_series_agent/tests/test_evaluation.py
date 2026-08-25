"""Tests for chronological forecast evaluation."""

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
    chronological_holdout,
    compute_forecast_metrics,
    create_holdout_comparison_plot,
    evaluate_models_on_holdout,
    save_holdout_results,
)
from time_series_agent.exceptions import EvaluationError


def make_series(
    observations: int = 240,
) -> pd.Series:
    """Create a deterministic hourly seasonal series."""
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
    """Return baseline factories for evaluation tests."""
    return {
        "mean": lambda: MeanForecaster(),
        "naive": lambda: NaiveForecaster(),
        "seasonal_24": lambda: SeasonalNaiveForecaster(24),
    }


def test_chronological_holdout_preserves_time_order() -> None:
    """Training must end before the hidden test period begins."""
    series = make_series(observations=10)

    training, test = chronological_holdout(
        series,
        horizon=3,
    )

    assert len(training) == 7
    assert len(test) == 3
    assert training.index.max() < test.index.min()
    assert test.equals(series.iloc[-3:])


@pytest.mark.parametrize("invalid_horizon", [0, -1, 10])
def test_invalid_holdout_horizon_is_rejected(
    invalid_horizon: int,
) -> None:
    """The horizon must be positive and smaller than the series."""
    series = make_series(observations=10)

    with pytest.raises(EvaluationError):
        chronological_holdout(
            series,
            horizon=invalid_horizon,
        )


def test_metrics_match_known_errors() -> None:
    """MAE, RMSE, and MASE should match a simple example."""
    training = pd.Series(
        [1, 2, 3, 4],
        index=pd.date_range(
            "2026-01-01",
            periods=4,
            freq="h",
        ),
        dtype="float64",
    )
    actual = pd.Series(
        [5, 7],
        index=pd.date_range(
            "2026-01-01 04:00:00",
            periods=2,
            freq="h",
        ),
        dtype="float64",
    )
    forecast = pd.Series(
        [4, 8],
        index=actual.index,
        dtype="float64",
    )

    metrics = compute_forecast_metrics(
        actual=actual,
        forecast=forecast,
        training_series=training,
        mase_period=1,
    )

    assert metrics.mae == pytest.approx(1.0)
    assert metrics.rmse == pytest.approx(1.0)
    assert metrics.mase == pytest.approx(1.0)
    assert 0 < metrics.smape < 200


def test_smape_handles_double_zero() -> None:
    """An actual and forecast pair of zero should contribute zero."""
    training = make_series(observations=48)
    test_index = pd.date_range(
        training.index[-1] + pd.Timedelta(1, unit="h"),
        periods=2,
        freq="h",
    )
    actual = pd.Series(
        [0, 0],
        index=test_index,
        dtype="float64",
    )
    forecast = pd.Series(
        [0, 2],
        index=test_index,
        dtype="float64",
    )

    metrics = compute_forecast_metrics(
        actual,
        forecast,
        training,
        mase_period=24,
    )

    assert metrics.smape == pytest.approx(100.0)


def test_metric_timestamps_must_match() -> None:
    """Actual and forecast timestamps must align exactly."""
    training = make_series(observations=48)
    actual = make_series(observations=2)
    forecast = actual.copy()
    forecast.index = forecast.index + pd.Timedelta(
        1,
        unit="h",
    )

    with pytest.raises(
        EvaluationError,
        match="timestamps must match",
    ):
        compute_forecast_metrics(
            actual,
            forecast,
            training,
            mase_period=24,
        )


def test_training_must_support_mase_period() -> None:
    """MASE requires more training rows than its seasonal period."""
    training = make_series(observations=24)
    actual = make_series(observations=2)
    forecast = actual.copy()

    with pytest.raises(
        EvaluationError,
        match="too short",
    ):
        compute_forecast_metrics(
            actual,
            forecast,
            training,
            mase_period=24,
        )


def test_holdout_evaluates_all_models() -> None:
    """Evaluation should return one metric row per model."""
    series = make_series()

    metrics, forecasts = evaluate_models_on_holdout(
        series=series,
        model_factories=model_factories(),
        horizon=24,
        mase_period=24,
    )

    assert set(metrics["model"]) == {
        "mean",
        "naive",
        "seasonal_24",
    }
    assert len(metrics) == 3
    assert forecasts.shape == (24, 4)
    assert forecasts.isna().sum().sum() == 0


def test_empty_model_mapping_is_rejected() -> None:
    """Evaluation requires at least one model."""
    with pytest.raises(
        EvaluationError,
        match="At least one model",
    ):
        evaluate_models_on_holdout(
            series=make_series(),
            model_factories={},
            horizon=24,
            mase_period=24,
        )


def test_results_and_plot_can_be_saved(
    tmp_path: Path,
) -> None:
    """Evaluation artifacts should be written successfully."""
    metrics, forecasts = evaluate_models_on_holdout(
        series=make_series(),
        model_factories=model_factories(),
        horizon=24,
        mase_period=24,
    )

    metrics_path = tmp_path / "metrics.csv"
    forecasts_path = tmp_path / "forecasts.csv"
    figure_path = tmp_path / "comparison.png"

    save_holdout_results(
        metrics=metrics,
        forecasts=forecasts,
        metrics_path=metrics_path,
        forecasts_path=forecasts_path,
    )
    create_holdout_comparison_plot(
        forecasts,
        figure_path,
    )

    assert metrics_path.is_file()
    assert forecasts_path.is_file()
    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0