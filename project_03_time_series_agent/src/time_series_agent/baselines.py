"""Simple, reproducible baseline forecasting models."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from time_series_agent.exceptions import (
    ForecastingError,
    ModelNotFittedError,
)


def _validate_training_series(
    training_series: pd.Series,
    frequency: str,
) -> pd.Series:
    """Validate and return a numeric, regularly spaced series."""
    if not isinstance(training_series, pd.Series):
        raise ForecastingError(
            "Training data must be supplied as a pandas Series."
        )

    if training_series.empty:
        raise ForecastingError(
            "Training series must contain at least one observation."
        )

    if not isinstance(training_series.index, pd.DatetimeIndex):
        raise ForecastingError(
            "Training series must use a pandas DatetimeIndex."
        )

    if training_series.index.has_duplicates:
        raise ForecastingError(
            "Training timestamps must be unique."
        )

    if not training_series.index.is_monotonic_increasing:
        raise ForecastingError(
            "Training timestamps must be chronologically ordered."
        )

    numeric_values = pd.to_numeric(
        training_series,
        errors="coerce",
    )

    if numeric_values.isna().any():
        raise ForecastingError(
            "Training series contains missing or nonnumeric values."
        )

    expected_index = pd.date_range(
        start=training_series.index[0],
        periods=len(training_series),
        freq=frequency,
    )

    if not training_series.index.equals(expected_index):
        raise ForecastingError(
            f"Training timestamps must follow a regular "
            f"'{frequency}' frequency."
        )

    return pd.Series(
        numeric_values.to_numpy(dtype="float64"),
        index=training_series.index.copy(),
        name=training_series.name,
    )


class BaseForecaster(ABC):
    """Common interface for baseline forecasting models."""

    model_name = "baseline"

    def __init__(self, frequency: str = "h") -> None:
        """Initialize a forecaster with its observation frequency."""
        try:
            self._offset = to_offset(frequency)
        except ValueError as error:
            raise ForecastingError(
                f"Invalid forecasting frequency: {frequency}"
            ) from error

        self.frequency = frequency
        self._last_timestamp: pd.Timestamp | None = None

    def fit(self, training_series: pd.Series) -> Self:
        """Fit the model to chronologically ordered observations."""
        validated = _validate_training_series(
            training_series,
            self.frequency,
        )

        self._fit_values(validated)
        self._last_timestamp = validated.index[-1]

        return self

    def predict(self, horizon: int) -> pd.Series:
        """Forecast a positive number of future observations."""
        if self._last_timestamp is None:
            raise ModelNotFittedError(
                f"{self.model_name} model must be fitted before prediction."
            )

        if (
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or horizon <= 0
        ):
            raise ForecastingError(
                "Forecast horizon must be a positive integer."
            )

        forecast_index = pd.date_range(
            start=self._last_timestamp + self._offset,
            periods=horizon,
            freq=self._offset,
        )

        forecast_values = self._forecast_values(horizon)

        return pd.Series(
            forecast_values,
            index=forecast_index,
            name=f"{self.model_name}_forecast",
            dtype="float64",
        )

    @abstractmethod
    def _fit_values(self, training_series: pd.Series) -> None:
        """Store model-specific fitted state."""

    @abstractmethod
    def _forecast_values(self, horizon: int) -> np.ndarray:
        """Return model-specific future values."""


class MeanForecaster(BaseForecaster):
    """Forecast every future value using the training mean."""

    model_name = "mean"

    def __init__(self, frequency: str = "h") -> None:
        """Initialize an unfitted mean forecaster."""
        super().__init__(frequency=frequency)
        self._mean: float | None = None

    def _fit_values(self, training_series: pd.Series) -> None:
        """Store the arithmetic mean of the training series."""
        self._mean = float(training_series.mean())

    def _forecast_values(self, horizon: int) -> np.ndarray:
        """Repeat the fitted training mean."""
        if self._mean is None:
            raise ModelNotFittedError(
                "Mean model has no fitted training mean."
            )

        return np.full(
            shape=horizon,
            fill_value=self._mean,
            dtype="float64",
        )


class NaiveForecaster(BaseForecaster):
    """Forecast every future value using the latest observation."""

    model_name = "naive"

    def __init__(self, frequency: str = "h") -> None:
        """Initialize an unfitted naïve forecaster."""
        super().__init__(frequency=frequency)
        self._last_value: float | None = None

    def _fit_values(self, training_series: pd.Series) -> None:
        """Store the final observed training value."""
        self._last_value = float(training_series.iloc[-1])

    def _forecast_values(self, horizon: int) -> np.ndarray:
        """Repeat the latest observed training value."""
        if self._last_value is None:
            raise ModelNotFittedError(
                "Naïve model has no fitted final value."
            )

        return np.full(
            shape=horizon,
            fill_value=self._last_value,
            dtype="float64",
        )


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat the most recent observed seasonal cycle."""

    model_name = "seasonal_naive"

    def __init__(
        self,
        seasonal_period: int,
        frequency: str = "h",
    ) -> None:
        """Initialize a seasonal-naïve model."""
        if (
            not isinstance(seasonal_period, int)
            or isinstance(seasonal_period, bool)
            or seasonal_period <= 0
        ):
            raise ForecastingError(
                "Seasonal period must be a positive integer."
            )

        super().__init__(frequency=frequency)

        self.seasonal_period = seasonal_period
        self._last_season: np.ndarray | None = None

    def _fit_values(self, training_series: pd.Series) -> None:
        """Store the most recent complete seasonal cycle."""
        if len(training_series) < self.seasonal_period:
            raise ForecastingError(
                f"Seasonal-naïve model requires at least "
                f"{self.seasonal_period} training observations."
            )

        self._last_season = training_series.tail(
            self.seasonal_period
        ).to_numpy(dtype="float64")

    def _forecast_values(self, horizon: int) -> np.ndarray:
        """Repeat the stored seasonal cycle through the horizon."""
        if self._last_season is None:
            raise ModelNotFittedError(
                "Seasonal-naïve model has no fitted seasonal cycle."
            )

        repetitions = (
            horizon + self.seasonal_period - 1
        ) // self.seasonal_period

        return np.tile(
            self._last_season,
            repetitions,
        )[:horizon]