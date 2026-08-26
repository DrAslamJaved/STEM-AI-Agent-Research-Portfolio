"""Leakage-safe collection of out-of-sample forecast residuals."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from time_series_agent.baselines import BaseForecaster
from time_series_agent.evaluation import (
    expanding_window_splits,
)
from time_series_agent.exceptions import EvaluationError


ModelFactory = Callable[[], BaseForecaster]


REQUIRED_RESIDUAL_COLUMNS = {
    "fold",
    "model",
    "training_start",
    "training_end",
    "test_start",
    "test_end",
    "timestamp",
    "actual",
    "forecast",
    "residual",
    "absolute_residual",
    "fold_raw_negative_forecast_count",
}


def _raw_negative_forecast_count(
    model: BaseForecaster,
) -> int:
    """Read a model's raw-negative counter when available."""
    counter = getattr(
        model,
        "last_raw_negative_forecast_count",
        None,
    )

    if not callable(counter):
        return 0

    value = counter()

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise EvaluationError(
            "The model returned an invalid raw-negative "
            "forecast count."
        )

    return value


def _validate_fold_forecast(
    forecast: pd.Series,
    test: pd.Series,
) -> pd.Series:
    """Validate one fold forecast against its test period."""
    if not isinstance(forecast, pd.Series):
        raise EvaluationError(
            "The model forecast must be a pandas Series."
        )

    if len(forecast) != len(test):
        raise EvaluationError(
            "The forecast length does not match the "
            "test-period length."
        )

    if not isinstance(forecast.index, pd.DatetimeIndex):
        raise EvaluationError(
            "The forecast must use a DatetimeIndex."
        )

    if not forecast.index.equals(test.index):
        raise EvaluationError(
            "Forecast timestamps do not match test timestamps."
        )

    numeric_forecast = pd.to_numeric(
        forecast,
        errors="coerce",
    )

    if numeric_forecast.isna().any():
        raise EvaluationError(
            "The forecast contains missing or nonnumeric values."
        )

    if not np.isfinite(
        numeric_forecast.to_numpy(dtype="float64")
    ).all():
        raise EvaluationError(
            "The forecast contains nonfinite values."
        )

    return numeric_forecast.astype("float64")


def collect_expanding_window_residuals(
    series: pd.Series,
    model_name: str,
    model_factory: ModelFactory,
    initial_train_size: int,
    horizon: int,
    step: int,
) -> pd.DataFrame:
    """Collect timestamp-level forecasts and residuals."""
    if (
        not isinstance(model_name, str)
        or not model_name.strip()
    ):
        raise EvaluationError(
            "'model_name' must be a nonempty string."
        )

    if not callable(model_factory):
        raise EvaluationError(
            "'model_factory' must be callable."
        )

    splits = expanding_window_splits(
        series=series,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step=step,
    )

    fold_frames: list[pd.DataFrame] = []

    for fold_number, training, test in splits:
        try:
            model = model_factory()
            forecast = model.fit(training).predict(horizon)
        except Exception as error:
            raise EvaluationError(
                f"Model '{model_name}' failed during "
                f"fold {fold_number}."
            ) from error

        validated_forecast = _validate_fold_forecast(
            forecast=forecast,
            test=test,
        )

        raw_negative_count = (
            _raw_negative_forecast_count(model)
        )

        actual_values = test.to_numpy(
            dtype="float64"
        )
        forecast_values = (
            validated_forecast.to_numpy(
                dtype="float64"
            )
        )
        residual_values = (
            actual_values - forecast_values
        )

        fold_frame = pd.DataFrame(
            {
                "fold": fold_number,
                "model": model_name,
                "training_start": str(
                    training.index.min()
                ),
                "training_end": str(
                    training.index.max()
                ),
                "test_start": str(
                    test.index.min()
                ),
                "test_end": str(
                    test.index.max()
                ),
                "timestamp": test.index,
                "actual": actual_values,
                "forecast": forecast_values,
                "residual": residual_values,
                "absolute_residual": np.abs(
                    residual_values
                ),
                "fold_raw_negative_forecast_count": (
                    raw_negative_count
                ),
            }
        )

        fold_frames.append(fold_frame)

    residuals = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    return residuals


def save_expanding_window_residuals(
    residuals: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save timestamp-level residual evidence to CSV."""
    if not isinstance(residuals, pd.DataFrame):
        raise EvaluationError(
            "'residuals' must be a pandas DataFrame."
        )

    if residuals.empty:
        raise EvaluationError(
            "The residual table cannot be empty."
        )

    missing_columns = sorted(
        REQUIRED_RESIDUAL_COLUMNS
        - set(residuals.columns)
    )

    if missing_columns:
        raise EvaluationError(
            "Residual table is missing required columns: "
            + ", ".join(missing_columns)
        )

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals.to_csv(
        destination,
        index=False,
    )

    return destination