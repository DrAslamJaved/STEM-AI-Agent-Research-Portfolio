"""Tests for classical statistical forecasting models."""

import numpy as np
import pandas as pd
import pytest

from time_series_agent.classical import HoltWintersForecaster
from time_series_agent.exceptions import (
    ForecastingError,
    ModelNotFittedError,
)


def make_seasonal_series(
    observations: int = 240,
    include_zeros: bool = False,
) -> pd.Series:
    """Create deterministic hourly data with daily seasonality."""
    time_index = np.arange(observations)

    values = (
        100
        + 0.08 * time_index
        + 20 * np.sin(2 * np.pi * time_index / 24)
    )

    if include_zeros:
        values = values.copy()
        values[::72] = 0

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


def test_holt_winters_produces_finite_forecast() -> None:
    """A fitted model should produce aligned finite forecasts."""
    model = HoltWintersForecaster(seasonal_period=24)
    training = make_seasonal_series()

    forecast = model.fit(training).predict(24)

    assert len(forecast) == 24
    assert np.isfinite(forecast).all()
    assert forecast.index[0] == (
        training.index[-1] + pd.Timedelta(1, unit="h")
    )


def test_holt_winters_is_deterministic() -> None:
    """Repeated fits on identical data should agree."""
    training = make_seasonal_series()

    first = (
        HoltWintersForecaster(24)
        .fit(training)
        .predict(24)
    )
    second = (
        HoltWintersForecaster(24)
        .fit(training)
        .predict(24)
    )

    np.testing.assert_allclose(
        first.to_numpy(),
        second.to_numpy(),
        rtol=1e-7,
        atol=1e-7,
    )


def test_additive_holt_winters_accepts_zeros() -> None:
    """Additive seasonality should support zero observations."""
    training = make_seasonal_series(
        include_zeros=True,
    )

    forecast = (
        HoltWintersForecaster(24)
        .fit(training)
        .predict(24)
    )

    assert np.isfinite(forecast).all()


def test_insufficient_seasonal_history_is_rejected() -> None:
    """At least two seasonal cycles are required."""
    training = make_seasonal_series(observations=47)

    with pytest.raises(
        ForecastingError,
        match="at least 48",
    ):
        HoltWintersForecaster(24).fit(training)


@pytest.mark.parametrize(
    "invalid_period",
    [0, 1, -24],
)
def test_invalid_seasonal_period_is_rejected(
    invalid_period: int,
) -> None:
    """Seasonal period must contain at least two observations."""
    with pytest.raises(
        ForecastingError,
        match="at least 2",
    ):
        HoltWintersForecaster(invalid_period)


def test_diagnostics_describe_fitted_residuals() -> None:
    """Diagnostics should be finite after model fitting."""
    model = HoltWintersForecaster(24)
    training = make_seasonal_series()

    model.fit(training)

    residuals = model.residuals()
    diagnostics = model.diagnostics()

    assert len(residuals) == len(training)
    assert diagnostics.training_rows == len(training)
    assert diagnostics.seasonal_period == 24
    assert np.isfinite(
        diagnostics.sum_squared_errors
    )
    assert np.isfinite(diagnostics.aic)
    assert np.isfinite(diagnostics.bic)
    assert np.isfinite(
        diagnostics.residual_standard_deviation
    )


def test_unfitted_diagnostics_are_rejected() -> None:
    """Diagnostics should require a fitted model."""
    model = HoltWintersForecaster(24)

    with pytest.raises(
        ModelNotFittedError,
        match="must be fitted",
    ):
        model.diagnostics()