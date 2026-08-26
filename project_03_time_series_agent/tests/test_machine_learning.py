"""Tests for recursive machine-learning forecasting."""

import numpy as np
import pandas as pd
import pytest

from time_series_agent.exceptions import (
    ForecastingError,
    ModelNotFittedError,
)
from time_series_agent.machine_learning import (
    RecursiveGradientBoostingForecaster,
)


def make_series(
    observations: int = 360,
    include_zeros: bool = False,
) -> pd.Series:
    """Create deterministic hourly seasonal data."""
    time_index = np.arange(observations)

    values = (
        150
        + 0.03 * time_index
        + 30 * np.sin(2 * np.pi * time_index / 24)
        + 10 * np.sin(2 * np.pi * time_index / 168)
    )

    if include_zeros:
        values = values.copy()
        values[::100] = 0

    return pd.Series(
        values,
        index=pd.date_range(
            "2026-01-01",
            periods=observations,
            freq="h",
        ),
        dtype="float64",
        name="target",
    )


def make_model() -> RecursiveGradientBoostingForecaster:
    """Create a small, fast model for unit tests."""
    return RecursiveGradientBoostingForecaster(
        n_estimators=50,
        learning_rate=0.05,
        random_state=42,
    )


def test_recursive_model_produces_aligned_forecast() -> None:
    """The model should produce finite future values."""
    training = make_series()
    model = make_model()

    forecast = model.fit(training).predict(24)

    assert len(forecast) == 24
    assert np.isfinite(forecast).all()
    assert forecast.ge(0).all()
    assert forecast.index[0] == (
        training.index[-1]
        + pd.Timedelta(1, unit="h")
    )


def test_recursive_model_is_deterministic() -> None:
    """A fixed random state should reproduce forecasts."""
    training = make_series()

    first = make_model().fit(training).predict(24)
    second = make_model().fit(training).predict(24)

    np.testing.assert_allclose(
        first.to_numpy(),
        second.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_recursive_model_accepts_zero_targets() -> None:
    """The tree model should support observed zeros."""
    forecast = (
        make_model()
        .fit(make_series(include_zeros=True))
        .predict(24)
    )

    assert forecast.ge(0).all()
    assert np.isfinite(forecast).all()


def test_diagnostics_report_feature_importance() -> None:
    """Diagnostics should contain aligned feature importance."""
    model = make_model()
    model.fit(make_series())

    diagnostics = model.diagnostics()

    assert diagnostics.feature_count == 12
    assert diagnostics.usable_training_rows == 192
    assert diagnostics.dropped_training_rows == 168
    assert len(diagnostics.feature_importances) == 12
    assert sum(
        diagnostics.feature_importances.values()
    ) == pytest.approx(1.0)
    assert diagnostics.training_mae >= 0


def test_insufficient_history_is_rejected() -> None:
    """At least 169 observations are required."""
    training = make_series(observations=168)

    with pytest.raises(
        ForecastingError,
        match="at least 169",
    ):
        make_model().fit(training)


def test_unfitted_prediction_is_rejected() -> None:
    """Prediction should require model fitting."""
    with pytest.raises(
        ModelNotFittedError,
        match="must be fitted",
    ):
        make_model().predict(24)


def test_invalid_estimator_count_is_rejected() -> None:
    """Estimator count must be positive."""
    with pytest.raises(
        ForecastingError,
        match="positive integer",
    ):
        RecursiveGradientBoostingForecaster(
            n_estimators=0
        )


def test_invalid_learning_rate_is_rejected() -> None:
    """Learning rate must be positive."""
    with pytest.raises(
        ForecastingError,
        match="must be positive",
    ):
        RecursiveGradientBoostingForecaster(
            learning_rate=0
        )