"""Reproducible exploratory analysis for time-series data."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from time_series_agent.exceptions import ExplorationError


@dataclass(frozen=True)
class ExplorationSummary:
    """Numerical summary of exploratory time-series analysis."""

    row_count: int
    start_timestamp: str
    end_timestamp: str
    target_minimum: float
    target_maximum: float
    target_mean: float
    target_median: float
    target_standard_deviation: float
    target_first_quartile: float
    target_third_quartile: float
    zero_target_count: int
    autocorrelation_lag_1: float
    autocorrelation_lag_24: float
    autocorrelation_lag_168: float
    adf_statistic: float
    adf_p_value: float
    adf_used_lags: int
    adf_observations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _prepare_series(
    data: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    minimum_observations: int = 48,
) -> pd.Series:
    """Create a validated, ordered numeric series for exploration."""
    missing_columns = {
        timestamp_column,
        target_column,
    }.difference(data.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ExplorationError(
            f"Exploration requires missing column(s): {missing_text}"
        )

    timestamps = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
    )
    target = pd.to_numeric(
        data[target_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ExplorationError(
            "Exploration cannot use invalid or missing timestamps."
        )

    if target.isna().any():
        raise ExplorationError(
            "Exploration cannot use invalid or missing target values."
        )

    series = pd.Series(
        target.to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(timestamps),
        name=target_column,
    ).sort_index()

    if series.index.duplicated().any():
        raise ExplorationError(
            "Exploration cannot use duplicate timestamps."
        )

    if len(series) < minimum_observations:
        raise ExplorationError(
            f"Exploration requires at least {minimum_observations} "
            f"observations; received {len(series)}."
        )

    return series


def compute_exploration_summary(
    data: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
) -> ExplorationSummary:
    """Calculate descriptive, autocorrelation, and ADF statistics."""
    series = _prepare_series(
        data=data,
        timestamp_column=timestamp_column,
        target_column=target_column,
        minimum_observations=48,
    )

    adf_result = adfuller(
        series,
        autolag="AIC",
    )

    return ExplorationSummary(
        row_count=len(series),
        start_timestamp=str(series.index.min()),
        end_timestamp=str(series.index.max()),
        target_minimum=float(series.min()),
        target_maximum=float(series.max()),
        target_mean=float(series.mean()),
        target_median=float(series.median()),
        target_standard_deviation=float(series.std()),
        target_first_quartile=float(series.quantile(0.25)),
        target_third_quartile=float(series.quantile(0.75)),
        zero_target_count=int(series.eq(0).sum()),
        autocorrelation_lag_1=float(series.autocorr(lag=1)),
        autocorrelation_lag_24=float(series.autocorr(lag=24)),
        autocorrelation_lag_168=float(series.autocorr(lag=168)),
        adf_statistic=float(adf_result[0]),
        adf_p_value=float(adf_result[1]),
        adf_used_lags=int(adf_result[2]),
        adf_observations=int(adf_result[3]),
    )


def _save_figure(
    figure: plt.Figure,
    path: Path,
) -> None:
    """Save and close one figure."""
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_exploration_figures(
    data: pd.DataFrame,
    timestamp_column: str,
    target_column: str,
    output_directory: str | Path,
    seasonal_period: int = 24,
    maximum_lag: int = 72,
) -> tuple[Path, ...]:
    """Create reproducible exploratory figures.

    Parameters
    ----------
    data:
        Processed time-series data.
    timestamp_column:
        Name of the canonical timestamp.
    target_column:
        Name of the forecasting target.
    output_directory:
        Directory in which PNG figures will be written.
    seasonal_period:
        Number of observations in the primary seasonal cycle.
    maximum_lag:
        Maximum lag displayed in ACF and PACF figures.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Paths of the five generated figures.
    """
    series = _prepare_series(
        data=data,
        timestamp_column=timestamp_column,
        target_column=target_column,
        minimum_observations=max(
            2 * seasonal_period,
            2 * maximum_lag + 1,
        ),
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_paths: list[Path] = []

    full_series_path = output_path / "01_hourly_demand_series.png"

    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(
        series.index,
        series.values,
        linewidth=0.6,
        color="#2455A4",
    )
    axis.set_title("Hourly Seoul bike rentals")
    axis.set_xlabel("Date")
    axis.set_ylabel("Rented bike count")
    axis.grid(alpha=0.25)

    _save_figure(figure, full_series_path)
    figure_paths.append(full_series_path)

    rolling_path = output_path / "02_rolling_statistics.png"
    rolling_mean_24 = series.rolling(
        window=24,
        min_periods=24,
    ).mean()
    rolling_mean_168 = series.rolling(
        window=168,
        min_periods=168,
    ).mean()
    rolling_std_168 = series.rolling(
        window=168,
        min_periods=168,
    ).std()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 8),
        sharex=True,
    )

    axes[0].plot(
        series.index,
        series.values,
        linewidth=0.35,
        alpha=0.35,
        color="#777777",
        label="Observed",
    )
    axes[0].plot(
        rolling_mean_24.index,
        rolling_mean_24.values,
        linewidth=1.0,
        color="#E07A1F",
        label="24-hour rolling mean",
    )
    axes[0].plot(
        rolling_mean_168.index,
        rolling_mean_168.values,
        linewidth=1.3,
        color="#2455A4",
        label="168-hour rolling mean",
    )
    axes[0].set_title("Rolling means")
    axes[0].set_ylabel("Rented bike count")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        rolling_std_168.index,
        rolling_std_168.values,
        linewidth=1.0,
        color="#8E3B8F",
    )
    axes[1].set_title("168-hour rolling standard deviation")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Standard deviation")
    axes[1].grid(alpha=0.25)

    _save_figure(figure, rolling_path)
    figure_paths.append(rolling_path)

    hourly_profile_path = (
        output_path / "03_average_demand_by_hour.png"
    )
    hourly_profile = series.groupby(series.index.hour).mean()

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        hourly_profile.index,
        hourly_profile.values,
        color="#2A9D8F",
    )
    axis.set_title("Average rented-bike demand by hour")
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Average rented bike count")
    axis.set_xticks(range(24))
    axis.grid(axis="y", alpha=0.25)

    _save_figure(figure, hourly_profile_path)
    figure_paths.append(hourly_profile_path)

    correlation_path = output_path / "04_acf_pacf.png"

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13, 8),
    )
    plot_acf(
        series,
        lags=maximum_lag,
        ax=axes[0],
        zero=False,
    )
    axes[0].set_title(
        f"Autocorrelation function through lag {maximum_lag}"
    )
    axes[0].set_xlabel("Lag in hours")
    axes[0].set_ylabel("Autocorrelation")

    plot_pacf(
        series,
        lags=maximum_lag,
        ax=axes[1],
        zero=False,
        method="ywm",
    )
    axes[1].set_title(
        f"Partial autocorrelation function through lag {maximum_lag}"
    )
    axes[1].set_xlabel("Lag in hours")
    axes[1].set_ylabel("Partial autocorrelation")

    _save_figure(figure, correlation_path)
    figure_paths.append(correlation_path)

    decomposition_path = (
        output_path / "05_daily_seasonal_decomposition.png"
    )
    decomposition = seasonal_decompose(
        series,
        model="additive",
        period=seasonal_period,
        extrapolate_trend="freq",
    )

    display_points = min(len(series), 14 * seasonal_period)
    components = [
        ("Observed", decomposition.observed.tail(display_points)),
        ("Trend", decomposition.trend.tail(display_points)),
        ("Daily seasonal", decomposition.seasonal.tail(display_points)),
        ("Residual", decomposition.resid.tail(display_points)),
    ]

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 10),
        sharex=True,
    )

    for axis, (label, component) in zip(axes, components):
        axis.plot(
            component.index,
            component.values,
            linewidth=0.8,
            color="#2455A4",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    axes[0].set_title(
        "Additive daily-seasonal decomposition: final 14 days"
    )
    axes[-1].set_xlabel("Date")

    _save_figure(figure, decomposition_path)
    figure_paths.append(decomposition_path)

    return tuple(figure_paths)


def save_exploration_summary(
    summary: ExplorationSummary,
    output_path: str | Path,
) -> None:
    """Save exploratory statistics as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )