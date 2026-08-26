"""Tests for out-of-sample residual collection."""

import numpy as np
import pandas as pd
import pytest

from time_series_agent.baselines import (
    SeasonalNaiveForecaster,
)
from time_series_agent.exceptions import EvaluationError
from time_series_agent.residuals import (
    collect_expanding_window_residuals,
    save_expanding_window_residuals,
)


def make_series(
    observations: int = 144,
) -> pd.Series:
    """Create deterministic hourly test data."""
    time_index = np.arange(observations)

    values = (
        100
        + 0.05 * time_index
        + 20 * np.sin(2 * np.pi * time_index / 24)
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


def seasonal_factory() -> SeasonalNaiveForecaster:
    """Create a daily seasonal-naive model."""
    return SeasonalNaiveForecaster(
        seasonal_period=24,
        frequency="h",
    )


def test_collects_timestamp_level_residuals() -> None:
    """Collector should preserve every unseen timestamp."""
    series = make_series()

    residuals = collect_expanding_window_residuals(
        series=series,
        model_name="seasonal_naive_24",
        model_factory=seasonal_factory,
        initial_train_size=72,
        horizon=24,
        step=24,
    )

    assert len(residuals) == 72
    assert residuals["fold"].nunique() == 3
    assert residuals["timestamp"].nunique() == 72
    assert residuals["timestamp"].is_monotonic_increasing

    assert residuals["timestamp"].iloc[0] == (
        series.index[72]
    )
    assert residuals["timestamp"].iloc[-1] == (
        series.index[-1]
    )

    np.testing.assert_allclose(
        residuals["residual"],
        residuals["actual"] - residuals["forecast"],
    )

    np.testing.assert_allclose(
        residuals["absolute_residual"],
        np.abs(residuals["residual"]),
    )

    assert residuals.isna().sum().sum() == 0
    assert residuals[
        "fold_raw_negative_forecast_count"
    ].eq(0).all()


def test_empty_model_name_is_rejected() -> None:
    """A model must have a reportable name."""
    with pytest.raises(
        EvaluationError,
        match="model_name",
    ):
        collect_expanding_window_residuals(
            series=make_series(),
            model_name="",
            model_factory=seasonal_factory,
            initial_train_size=72,
            horizon=24,
            step=24,
        )


def test_noncallable_factory_is_rejected() -> None:
    """The collector should require a model factory."""
    with pytest.raises(
        EvaluationError,
        match="model_factory",
    ):
        collect_expanding_window_residuals(
            series=make_series(),
            model_name="bad_model",
            model_factory=None,  # type: ignore[arg-type]
            initial_train_size=72,
            horizon=24,
            step=24,
        )


class MisalignedForecastModel:
    """Test model that deliberately shifts forecast timestamps."""

    def fit(
        self,
        training_series: pd.Series,
    ) -> "MisalignedForecastModel":
        """Remember the final training timestamp."""
        self.training_end = training_series.index[-1]
        return self

    def predict(
        self,
        horizon: int,
    ) -> pd.Series:
        """Return deliberately misaligned predictions."""
        forecast_index = pd.date_range(
            start=self.training_end,
            periods=horizon + 2,
            freq="h",
        )[2:]

        return pd.Series(
            np.ones(horizon),
            index=forecast_index,
            dtype="float64",
        )


def test_misaligned_forecast_is_rejected() -> None:
    """Forecast timestamps must equal test timestamps."""
    with pytest.raises(
        EvaluationError,
        match="timestamps",
    ):
        collect_expanding_window_residuals(
            series=make_series(),
            model_name="misaligned",
            model_factory=MisalignedForecastModel,
            initial_train_size=72,
            horizon=24,
            step=24,
        )


def test_residual_table_can_be_saved(
    tmp_path,
) -> None:
    """Validated residual evidence should be saveable."""
    residuals = collect_expanding_window_residuals(
        series=make_series(),
        model_name="seasonal_naive_24",
        model_factory=seasonal_factory,
        initial_train_size=72,
        horizon=24,
        step=24,
    )

    output_path = (
        tmp_path / "nested" / "residuals.csv"
    )

    saved_path = save_expanding_window_residuals(
        residuals=residuals,
        output_path=output_path,
    )

    loaded = pd.read_csv(saved_path)

    assert saved_path.exists()
    assert len(loaded) == len(residuals)
    assert set(residuals.columns) == set(loaded.columns)