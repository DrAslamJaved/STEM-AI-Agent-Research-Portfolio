"""Time-aware forecasting evaluation utilities."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_series_agent.baselines import BaseForecaster
from time_series_agent.exceptions import EvaluationError


@dataclass(frozen=True)
class ForecastMetrics:
    """Accuracy metrics for one model and test period."""

    mae: float
    rmse: float
    smape: float
    mase: float

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""
        return asdict(self)


ModelFactory = Callable[[], BaseForecaster]


def _validate_evaluation_series(
    series: pd.Series,
) -> pd.Series:
    """Validate a complete series before temporal splitting."""
    if not isinstance(series, pd.Series):
        raise EvaluationError(
            "Evaluation data must be a pandas Series."
        )

    if series.empty:
        raise EvaluationError(
            "Evaluation series must not be empty."
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        raise EvaluationError(
            "Evaluation series must use a DatetimeIndex."
        )

    if series.index.has_duplicates:
        raise EvaluationError(
            "Evaluation timestamps must be unique."
        )

    if not series.index.is_monotonic_increasing:
        raise EvaluationError(
            "Evaluation timestamps must be chronologically ordered."
        )

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        raise EvaluationError(
            "Evaluation series contains missing or nonnumeric values."
        )

    return pd.Series(
        numeric.to_numpy(dtype="float64"),
        index=series.index.copy(),
        name=series.name,
    )


def chronological_holdout(
    series: pd.Series,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    """Split a series into past training and future test portions."""
    validated = _validate_evaluation_series(series)

    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon <= 0
    ):
        raise EvaluationError(
            "Holdout horizon must be a positive integer."
        )

    if horizon >= len(validated):
        raise EvaluationError(
            "Holdout horizon must be smaller than the series length."
        )

    training = validated.iloc[:-horizon].copy()
    test = validated.iloc[-horizon:].copy()

    if training.index.max() >= test.index.min():
        raise EvaluationError(
            "Training and test periods are not chronologically separated."
        )

    return training, test


def compute_forecast_metrics(
    actual: pd.Series,
    forecast: pd.Series,
    training_series: pd.Series,
    mase_period: int,
) -> ForecastMetrics:
    """Calculate MAE, RMSE, sMAPE, and MASE.

    The MASE denominator is calculated only from the training series.
    This prevents information from the test period entering the scale.
    """
    actual_values = _validate_evaluation_series(actual)
    forecast_values = _validate_evaluation_series(forecast)
    training_values = _validate_evaluation_series(training_series)

    if not actual_values.index.equals(forecast_values.index):
        raise EvaluationError(
            "Actual and forecast timestamps must match exactly."
        )

    if (
        not isinstance(mase_period, int)
        or isinstance(mase_period, bool)
        or mase_period <= 0
    ):
        raise EvaluationError(
            "MASE period must be a positive integer."
        )

    if len(training_values) <= mase_period:
        raise EvaluationError(
            "Training series is too short for the MASE period."
        )

    errors = (
        actual_values.to_numpy()
        - forecast_values.to_numpy()
    )

    absolute_errors = np.abs(errors)

    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    actual_array = actual_values.to_numpy()
    forecast_array = forecast_values.to_numpy()
    smape_denominator = (
        np.abs(actual_array) + np.abs(forecast_array)
    )

    smape_terms = np.divide(
        absolute_errors,
        smape_denominator,
        out=np.zeros_like(
            absolute_errors,
            dtype="float64",
        ),
        where=smape_denominator != 0,
    )
    smape = float(200 * np.mean(smape_terms))

    training_array = training_values.to_numpy()
    seasonal_differences = np.abs(
        training_array[mase_period:]
        - training_array[:-mase_period]
    )
    mase_scale = float(np.mean(seasonal_differences))

    if not np.isfinite(mase_scale) or mase_scale <= 0:
        raise EvaluationError(
            "MASE scale must be finite and greater than zero."
        )

    mase = float(mae / mase_scale)

    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        smape=smape,
        mase=mase,
    )


def evaluate_models_on_holdout(
    series: pd.Series,
    model_factories: Mapping[str, ModelFactory],
    horizon: int,
    mase_period: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and evaluate models on one chronological holdout."""
    if not model_factories:
        raise EvaluationError(
            "At least one model factory is required."
        )

    training, test = chronological_holdout(
        series=series,
        horizon=horizon,
    )

    metric_rows: list[dict[str, Any]] = []
    forecast_table = pd.DataFrame(
        {"actual": test},
        index=test.index,
    )

    for model_name, factory in model_factories.items():
        model = factory()
        forecast = model.fit(training).predict(horizon)

        metrics = compute_forecast_metrics(
            actual=test,
            forecast=forecast,
            training_series=training,
            mase_period=mase_period,
        )

        forecast_table[model_name] = forecast

        metric_rows.append(
            {
                "model": model_name,
                "training_start": str(training.index.min()),
                "training_end": str(training.index.max()),
                "test_start": str(test.index.min()),
                "test_end": str(test.index.max()),
                "training_rows": len(training),
                "test_rows": len(test),
                **metrics.to_dict(),
            }
        )

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table = metrics_table.sort_values(
        by=["mae", "rmse"],
        ascending=True,
    ).reset_index(drop=True)

    forecast_table.index.name = "timestamp"

    return metrics_table, forecast_table


def save_holdout_results(
    metrics: pd.DataFrame,
    forecasts: pd.DataFrame,
    metrics_path: str | Path,
    forecasts_path: str | Path,
) -> None:
    """Save holdout metrics and forecasts as CSV files."""
    metrics_output = Path(metrics_path)
    forecasts_output = Path(forecasts_path)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    forecasts_output.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        metrics_output,
        index=False,
        encoding="utf-8",
    )
    forecasts.to_csv(
        forecasts_output,
        index=True,
        encoding="utf-8",
    )


def create_holdout_comparison_plot(
    forecasts: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot actual values alongside all holdout forecasts."""
    if "actual" not in forecasts.columns:
        raise EvaluationError(
            "Forecast table must contain an 'actual' column."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(15, 6))

    axis.plot(
        forecasts.index,
        forecasts["actual"],
        color="#111111",
        linewidth=2.0,
        label="Actual",
    )

    model_columns = [
        column
        for column in forecasts.columns
        if column != "actual"
    ]

    for column in model_columns:
        axis.plot(
            forecasts.index,
            forecasts[column],
            linewidth=1.0,
            alpha=0.8,
            label=column,
        )

    axis.set_title(
        "Baseline forecasts on the final 168-hour holdout"
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Rented bike count")
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

def expanding_window_splits(
    series: pd.Series,
    initial_train_size: int,
    horizon: int,
    step: int,
) -> tuple[tuple[int, pd.Series, pd.Series], ...]:
    """Create non-random expanding-window temporal splits."""
    validated = _validate_evaluation_series(series)

    parameters = {
        "initial_train_size": initial_train_size,
        "horizon": horizon,
        "step": step,
    }

    for parameter_name, value in parameters.items():
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise EvaluationError(
                f"'{parameter_name}' must be a positive integer."
            )

    if initial_train_size + horizon > len(validated):
        raise EvaluationError(
            "Initial training size plus horizon exceeds "
            "the available series length."
        )

    splits: list[tuple[int, pd.Series, pd.Series]] = []
    fold_number = 1
    training_end = initial_train_size

    while training_end + horizon <= len(validated):
        training = validated.iloc[:training_end].copy()
        test = validated.iloc[
            training_end:training_end + horizon
        ].copy()

        if training.index.max() >= test.index.min():
            raise EvaluationError(
                "Expanding-window split contains temporal leakage."
            )

        splits.append(
            (
                fold_number,
                training,
                test,
            )
        )

        fold_number += 1
        training_end += step

    return tuple(splits)


def evaluate_models_expanding_window(
    series: pd.Series,
    model_factories: Mapping[str, ModelFactory],
    initial_train_size: int,
    horizon: int,
    step: int,
    mase_period: int,
) -> pd.DataFrame:
    """Evaluate models across multiple expanding temporal windows."""
    if not model_factories:
        raise EvaluationError(
            "At least one model factory is required."
        )

    splits = expanding_window_splits(
        series=series,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step=step,
    )

    result_rows: list[dict[str, Any]] = []

    for fold_number, training, test in splits:
        for model_name, factory in model_factories.items():
            model = factory()
            forecast = model.fit(training).predict(horizon)

            metrics = compute_forecast_metrics(
                actual=test,
                forecast=forecast,
                training_series=training,
                mase_period=mase_period,
            )

            result_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "training_start": str(
                        training.index.min()
                    ),
                    "training_end": str(
                        training.index.max()
                    ),
                    "test_start": str(test.index.min()),
                    "test_end": str(test.index.max()),
                    "training_rows": len(training),
                    "test_rows": len(test),
                    **metrics.to_dict(),
                }
            )

    return pd.DataFrame(result_rows)


def summarize_expanding_window_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate accuracy and variability across folds."""
    required_columns = {
        "fold",
        "model",
        "mae",
        "rmse",
        "smape",
        "mase",
    }
    missing_columns = required_columns.difference(
        fold_results.columns
    )

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise EvaluationError(
            f"Cross-validation results are missing: {missing_text}"
        )

    if fold_results.empty:
        raise EvaluationError(
            "Cross-validation results must not be empty."
        )

    summary = (
        fold_results.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_smape=("smape", "mean"),
            std_smape=("smape", "std"),
            mean_mase=("mase", "mean"),
            std_mase=("mase", "std"),
        )
    )

    standard_deviation_columns = [
        "std_mae",
        "std_rmse",
        "std_smape",
        "std_mase",
    ]
    summary[standard_deviation_columns] = summary[
        standard_deviation_columns
    ].fillna(0.0)

    winning_rows = fold_results.loc[
        fold_results.groupby("fold")["mae"].idxmin()
    ]
    fold_wins = winning_rows.groupby("model").size()

    summary["mae_fold_wins"] = (
        summary["model"]
        .map(fold_wins)
        .fillna(0)
        .astype(int)
    )

    summary["mae_rank"] = (
        summary["mean_mae"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return summary.sort_values(
        by=["mae_rank", "mean_rmse"],
        ascending=True,
    ).reset_index(drop=True)


def save_expanding_window_results(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    fold_results_path: str | Path,
    summary_path: str | Path,
) -> None:
    """Save fold-level and aggregate cross-validation results."""
    fold_output = Path(fold_results_path)
    summary_output = Path(summary_path)

    fold_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    fold_results.to_csv(
        fold_output,
        index=False,
        encoding="utf-8",
    )
    summary.to_csv(
        summary_output,
        index=False,
        encoding="utf-8",
    )


def create_expanding_window_plots(
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Create fold-level and aggregate MAE comparison plots."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_figure_path = (
        output_path / "07_expanding_window_mae_by_fold.png"
    )
    summary_figure_path = (
        output_path / "08_expanding_window_mean_mae.png"
    )

    figure, axis = plt.subplots(figsize=(13, 6))

    for model_name, model_results in fold_results.groupby(
        "model"
    ):
        ordered = model_results.sort_values("fold")

        axis.plot(
            ordered["fold"],
            ordered["mae"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=model_name,
        )

    axis.set_title("Baseline MAE across expanding weekly folds")
    axis.set_xlabel("Fold")
    axis.set_ylabel("Mean absolute error")
    axis.set_xticks(
        sorted(fold_results["fold"].unique())
    )
    axis.legend(ncol=2)
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        fold_figure_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    ordered_summary = summary.sort_values(
        "mean_mae",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        ordered_summary["model"],
        ordered_summary["mean_mae"],
        yerr=ordered_summary["std_mae"],
        capsize=5,
        color="#2A9D8F",
    )
    axis.set_title(
        "Mean baseline MAE across expanding weekly folds"
    )
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean absolute error")
    axis.tick_params(
        axis="x",
        rotation=20,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()
    figure.savefig(
        summary_figure_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    return fold_figure_path, summary_figure_path