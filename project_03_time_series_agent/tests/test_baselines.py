"""Tests for baseline forecasting models."""

import pandas as pd
import pytest

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.exceptions import (
    ForecastingError,
    ModelNotFittedError,
)


def make_series(
    values: list[float],
) -> pd.Series:
    """Create a regular hourly training series."""
    return pd.Series(
        values,
        index=pd.date_range(
            "2026-01-01 00:00:00",
            periods=len(values),
            freq="h",
        ),
        dtype="float64",
    )


def test_mean_forecaster_repeats_training_mean() -> None:
    """Mean forecasts should equal the training mean."""
    model = MeanForecaster()
    model.fit(make_series([10, 20, 30, 40]))

    forecast = model.predict(3)

    assert forecast.tolist() == [25, 25, 25]
    assert forecast.index[0] == pd.Timestamp(
        "2026-01-01 04:00:00"
    )


def test_naive_forecaster_repeats_last_value() -> None:
    """Naïve forecasts should equal the latest observation."""
    model = NaiveForecaster()
    model.fit(make_series([10, 20, 30, 40]))

    forecast = model.predict(3)

    assert forecast.tolist() == [40, 40, 40]


def test_seasonal_naive_repeats_last_cycle() -> None:
    """Seasonal forecasts should repeat the latest cycle."""
    model = SeasonalNaiveForecaster(seasonal_period=2)
    model.fit(make_series([10, 20, 30, 40]))

    forecast = model.predict(5)

    assert forecast.tolist() == [30, 40, 30, 40, 30]


@pytest.mark.parametrize(
    "model",
    [
        MeanForecaster(),
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonal_period=2),
    ],
)
def test_prediction_before_fit_is_rejected(
    model: object,
) -> None:
    """Every model should require fitting before prediction."""
    with pytest.raises(
        ModelNotFittedError,
        match="must be fitted",
    ):
        model.predict(2)  # type: ignore[attr-defined]


@pytest.mark.parametrize("invalid_horizon", [0, -1, 1.5])
def test_invalid_forecast_horizon_is_rejected(
    invalid_horizon: object,
) -> None:
    """Forecast horizons must be positive integers."""
    model = MeanForecaster().fit(
        make_series([10, 20, 30])
    )

    with pytest.raises(
        ForecastingError,
        match="positive integer",
    ):
        model.predict(invalid_horizon)  # type: ignore[arg-type]


def test_missing_training_value_is_rejected() -> None:
    """Baseline models should reject missing training targets."""
    series = make_series([10, 20, 30])
    series.iloc[1] = None

    with pytest.raises(
        ForecastingError,
        match="missing or nonnumeric",
    ):
        MeanForecaster().fit(series)


def test_non_datetime_index_is_rejected() -> None:
    """Training data must have a temporal index."""
    series = pd.Series(
        [10, 20, 30],
        dtype="float64",
    )

    with pytest.raises(
        ForecastingError,
        match="DatetimeIndex",
    ):
        NaiveForecaster().fit(series)


def test_irregular_timestamps_are_rejected() -> None:
    """A baseline should reject gaps in its training timeline."""
    series = pd.Series(
        [10, 20, 30],
        index=pd.DatetimeIndex(
            [
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
                "2026-01-01 03:00:00",
            ]
        ),
        dtype="float64",
    )

    with pytest.raises(
        ForecastingError,
        match="regular",
    ):
        MeanForecaster().fit(series)


def test_insufficient_seasonal_history_is_rejected() -> None:
    """Seasonal fitting requires one complete cycle."""
    model = SeasonalNaiveForecaster(seasonal_period=24)

    with pytest.raises(
        ForecastingError,
        match="at least 24",
    ):
        model.fit(make_series([10, 20, 30]))