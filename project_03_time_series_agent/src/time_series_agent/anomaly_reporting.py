"""Context enrichment and anomaly-episode construction."""

from pathlib import Path

import matplotlib.pyplot as plt

from collections.abc import Sequence

import numpy as np
import pandas as pd

from time_series_agent.exceptions import (
    AnomalyDetectionError,
)


LABEL_COLUMNS = {
    "timestamp",
    "actual",
    "forecast",
    "residual",
    "absolute_modified_z_score",
    "is_actionable_anomaly",
    "anomaly_type",
    "anomaly_severity",
}

SOURCE_COLUMNS = {
    "Humidity(%)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons",
    "Holiday",
    "Functioning Day",
}

EPISODE_COLUMNS = [
    "episode_id",
    "start_timestamp",
    "end_timestamp",
    "duration_hours",
    "anomaly_hours",
    "episode_direction",
    "episode_context",
    "maximum_severity",
    "peak_timestamp",
    "maximum_absolute_modified_z_score",
    "actual_total",
    "forecast_total",
    "total_residual",
    "forecast_to_actual_ratio",
    "minimum_forecast",
    "rainfall_total_mm",
    "maximum_hourly_rainfall_mm",
    "mean_temperature_c",
    "mean_humidity_percent",
    "holiday_values",
    "functioning_day_values",
]


def _missing_columns(
    frame: pd.DataFrame,
    required: set[str],
) -> list[str]:
    """Return sorted missing column names."""
    return sorted(required - set(frame.columns))


def enrich_anomaly_context(
    labels: pd.DataFrame,
    source_data: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Attach weather and operating context to anomaly labels."""
    if not isinstance(labels, pd.DataFrame):
        raise AnomalyDetectionError(
            "'labels' must be a pandas DataFrame."
        )

    if not isinstance(source_data, pd.DataFrame):
        raise AnomalyDetectionError(
            "'source_data' must be a pandas DataFrame."
        )

    if labels.empty or source_data.empty:
        raise AnomalyDetectionError(
            "Labels and source data must not be empty."
        )

    missing_labels = _missing_columns(
        labels,
        LABEL_COLUMNS,
    )

    if missing_labels:
        raise AnomalyDetectionError(
            "Anomaly labels are missing columns: "
            + ", ".join(missing_labels)
        )

    required_source = SOURCE_COLUMNS | {
        timestamp_column
    }
    missing_source = _missing_columns(
        source_data,
        required_source,
    )

    if missing_source:
        raise AnomalyDetectionError(
            "Source data is missing context columns: "
            + ", ".join(missing_source)
        )

    temperature_columns = [
        column
        for column in source_data.columns
        if str(column).startswith("Temperature")
    ]

    if len(temperature_columns) != 1:
        raise AnomalyDetectionError(
            "Exactly one temperature column is required."
        )

    temperature_column = temperature_columns[0]

    labels_copy = labels.copy(deep=True)
    source_copy = source_data.copy(deep=True)

    labels_copy["timestamp"] = pd.to_datetime(
        labels_copy["timestamp"],
        errors="coerce",
    )
    source_copy[timestamp_column] = pd.to_datetime(
        source_copy[timestamp_column],
        errors="coerce",
    )

    if labels_copy["timestamp"].isna().any():
        raise AnomalyDetectionError(
            "Labels contain invalid timestamps."
        )

    if source_copy[timestamp_column].isna().any():
        raise AnomalyDetectionError(
            "Source data contains invalid timestamps."
        )

    if labels_copy["timestamp"].duplicated().any():
        raise AnomalyDetectionError(
            "Anomaly-label timestamps must be unique."
        )

    if source_copy[timestamp_column].duplicated().any():
        raise AnomalyDetectionError(
            "Source-data timestamps must be unique."
        )

    if not pd.api.types.is_bool_dtype(
        labels_copy["is_actionable_anomaly"].dtype
    ):
        raise AnomalyDetectionError(
            "'is_actionable_anomaly' must be Boolean."
        )

    context_source = source_copy[
        [
            timestamp_column,
            temperature_column,
            "Humidity(%)",
            "Rainfall(mm)",
            "Snowfall (cm)",
            "Seasons",
            "Holiday",
            "Functioning Day",
        ]
    ].rename(
        columns={
            timestamp_column: "timestamp",
            temperature_column: "temperature_c",
            "Humidity(%)": "humidity_percent",
            "Rainfall(mm)": "rainfall_mm",
            "Snowfall (cm)": "snowfall_cm",
            "Seasons": "season",
            "Holiday": "holiday",
            "Functioning Day": "functioning_day",
        }
    )

    enriched = labels_copy.merge(
        context_source,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    added_columns = [
        "temperature_c",
        "humidity_percent",
        "rainfall_mm",
        "snowfall_cm",
        "season",
        "holiday",
        "functioning_day",
    ]

    if enriched[added_columns].isna().any().any():
        raise AnomalyDetectionError(
            "Context is missing for one or more anomaly rows."
        )

    return enriched.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def _joined_unique(
    values: Sequence[object],
) -> str:
    """Join unique context values deterministically."""
    return ", ".join(
        sorted(
            {
                str(value)
                for value in values
            }
        )
    )


def build_anomaly_episodes(
    contextual_labels: pd.DataFrame,
    maximum_gap_hours: int = 1,
    forecast_floor_threshold: float = 50.0,
) -> pd.DataFrame:
    """Group consecutive actionable hours into episodes."""
    if not isinstance(
        contextual_labels,
        pd.DataFrame,
    ):
        raise AnomalyDetectionError(
            "'contextual_labels' must be a DataFrame."
        )

    required_columns = LABEL_COLUMNS | {
        "temperature_c",
        "humidity_percent",
        "rainfall_mm",
        "holiday",
        "functioning_day",
    }

    missing_columns = _missing_columns(
        contextual_labels,
        required_columns,
    )

    if missing_columns:
        raise AnomalyDetectionError(
            "Contextual labels are missing columns: "
            + ", ".join(missing_columns)
        )

    if (
        not isinstance(maximum_gap_hours, int)
        or isinstance(maximum_gap_hours, bool)
        or maximum_gap_hours <= 0
    ):
        raise AnomalyDetectionError(
            "'maximum_gap_hours' must be a "
            "positive integer."
        )

    if (
        not isinstance(
            forecast_floor_threshold,
            (int, float),
        )
        or isinstance(
            forecast_floor_threshold,
            bool,
        )
        or not np.isfinite(
            forecast_floor_threshold
        )
        or forecast_floor_threshold < 0
    ):
        raise AnomalyDetectionError(
            "'forecast_floor_threshold' must be a "
            "nonnegative finite number."
        )

    actionable = contextual_labels.loc[
        contextual_labels[
            "is_actionable_anomaly"
        ]
    ].copy()

    if actionable.empty:
        return pd.DataFrame(
            columns=EPISODE_COLUMNS
        )

    actionable = actionable.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    gap_hours = (
        actionable["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    new_episode = (
        gap_hours.isna()
        | gap_hours.gt(maximum_gap_hours)
    )

    actionable["_episode_number"] = (
        new_episode.cumsum()
    )

    rows: list[dict[str, object]] = []

    for episode_number, group in actionable.groupby(
        "_episode_number",
        sort=True,
    ):
        group = group.copy()

        start = group["timestamp"].min()
        end = group["timestamp"].max()

        positive_count = int(
            group["anomaly_type"].eq(
                "positive_demand_spike"
            ).sum()
        )
        negative_count = int(
            group["anomaly_type"].eq(
                "negative_demand_drop"
            ).sum()
        )

        if positive_count and negative_count:
            episode_direction = "mixed"
        elif positive_count:
            episode_direction = "positive"
        else:
            episode_direction = "negative"

        rainfall_total = float(
            group["rainfall_mm"].sum()
        )

        minimum_forecast = float(
            group["forecast"].min()
        )

        if (
            episode_direction == "negative"
            and rainfall_total > 0
        ):
            episode_context = (
                "rain_coincident_negative_episode"
            )
        elif (
            episode_direction == "positive"
            and minimum_forecast
            <= forecast_floor_threshold
        ):
            episode_context = (
                "forecast_floor_positive_episode"
            )
        else:
            episode_context = (
                "other_residual_episode"
            )

        peak_index = group[
            "absolute_modified_z_score"
        ].idxmax()
        peak_row = group.loc[peak_index]

        actual_total = float(
            group["actual"].sum()
        )
        forecast_total = float(
            group["forecast"].sum()
        )

        if actual_total == 0:
            forecast_ratio = float("nan")
        else:
            forecast_ratio = float(
                forecast_total / actual_total
            )

        duration_hours = int(
            (
                end.to_pydatetime()
                - start.to_pydatetime()
            ).total_seconds()
            // 3600
        ) + 1

        rows.append(
            {
                "episode_id": int(episode_number),
                "start_timestamp": start,
                "end_timestamp": end,
                "duration_hours": duration_hours,
                "anomaly_hours": int(len(group)),
                "episode_direction": episode_direction,
                "episode_context": episode_context,
                "maximum_severity": str(
                    peak_row["anomaly_severity"]
                ),
                "peak_timestamp": peak_row[
                    "timestamp"
                ],
                "maximum_absolute_modified_z_score": float(
                    peak_row[
                        "absolute_modified_z_score"
                    ]
                ),
                "actual_total": actual_total,
                "forecast_total": forecast_total,
                "total_residual": float(
                    group["residual"].sum()
                ),
                "forecast_to_actual_ratio": (
                    forecast_ratio
                ),
                "minimum_forecast": minimum_forecast,
                "rainfall_total_mm": rainfall_total,
                "maximum_hourly_rainfall_mm": float(
                    group["rainfall_mm"].max()
                ),
                "mean_temperature_c": float(
                    group["temperature_c"].mean()
                ),
                "mean_humidity_percent": float(
                    group[
                        "humidity_percent"
                    ].mean()
                ),
                "holiday_values": _joined_unique(
                    group["holiday"]
                ),
                "functioning_day_values": (
                    _joined_unique(
                        group["functioning_day"]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=EPISODE_COLUMNS,
    )


def select_top_actionable_anomalies(
    contextual_labels: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    """Select the strongest actionable anomaly hours."""
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit <= 0
    ):
        raise AnomalyDetectionError(
            "'limit' must be a positive integer."
        )

    required_columns = LABEL_COLUMNS | {
        "temperature_c",
        "humidity_percent",
        "rainfall_mm",
        "holiday",
        "functioning_day",
    }

    missing_columns = _missing_columns(
        contextual_labels,
        required_columns,
    )

    if missing_columns:
        raise AnomalyDetectionError(
            "Contextual labels are missing columns: "
            + ", ".join(missing_columns)
        )

    selected = (
        contextual_labels.loc[
            contextual_labels[
                "is_actionable_anomaly"
            ]
        ]
        .sort_values(
            "absolute_modified_z_score",
            ascending=False,
        )
        .head(limit)
        .copy()
    )

    selected.insert(
        0,
        "anomaly_rank",
        range(1, len(selected) + 1),
    )

    return selected.reset_index(drop=True)

def create_anomaly_report_figures(
    contextual_labels: pd.DataFrame,
    output_directory: str | Path,
    threshold: float = 3.5,
) -> tuple[Path, Path, Path]:
    """Create full-resolution anomaly-report figures."""
    required_columns = LABEL_COLUMNS | {
        "is_known_closure",
        "modified_z_score",
    }

    missing_columns = _missing_columns(
        contextual_labels,
        required_columns,
    )

    if missing_columns:
        raise AnomalyDetectionError(
            "Figure data is missing columns: "
            + ", ".join(missing_columns)
        )

    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise AnomalyDetectionError(
            "'threshold' must be a positive finite number."
        )

    data = contextual_labels.copy(deep=True)
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():
        raise AnomalyDetectionError(
            "Figure data contains invalid timestamps."
        )

    output = Path(output_directory)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive = data["anomaly_type"].eq(
        "positive_demand_spike"
    )
    negative = data["anomaly_type"].eq(
        "negative_demand_drop"
    )
    closures = data["is_known_closure"]

    timeline_path = (
        output / "13_anomaly_timeline.png"
    )

    figure, axis = plt.subplots(
        figsize=(16, 6)
    )

    axis.plot(
        data["timestamp"],
        data["actual"],
        label="Actual demand",
        linewidth=1.0,
        color="#1f77b4",
    )
    axis.plot(
        data["timestamp"],
        data["forecast"],
        label="Gradient Boosting forecast",
        linewidth=0.9,
        color="#ff7f0e",
        alpha=0.85,
    )
    axis.scatter(
        data.loc[positive, "timestamp"],
        data.loc[positive, "actual"],
        label="Positive candidate",
        color="#d62728",
        marker="^",
        s=28,
        zorder=4,
    )
    axis.scatter(
        data.loc[negative, "timestamp"],
        data.loc[negative, "actual"],
        label="Negative candidate",
        color="#2ca02c",
        marker="v",
        s=28,
        zorder=4,
    )
    axis.scatter(
        data.loc[closures, "timestamp"],
        data.loc[closures, "actual"],
        label="Known closure",
        color="#7f7f7f",
        marker="x",
        s=18,
        alpha=0.7,
        zorder=3,
    )

    axis.set_title(
        "Out-of-Sample Forecasts and Residual Anomalies"
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Rented bike count")
    axis.legend(
        loc="upper left",
        ncol=3,
    )
    axis.grid(
        alpha=0.2,
    )

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(
        timeline_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    score_path = (
        output / "14_modified_z_score_timeline.png"
    )

    figure, axis = plt.subplots(
        figsize=(16, 5)
    )

    axis.plot(
        data["timestamp"],
        data["modified_z_score"],
        linewidth=0.8,
        color="#4c4c4c",
        alpha=0.8,
        label="Modified z-score",
    )
    axis.axhline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label=f"Upper threshold (+{threshold})",
    )
    axis.axhline(
        -threshold,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.2,
        label=f"Lower threshold (-{threshold})",
    )
    axis.scatter(
        data.loc[positive, "timestamp"],
        data.loc[positive, "modified_z_score"],
        color="#d62728",
        marker="^",
        s=25,
        zorder=4,
    )
    axis.scatter(
        data.loc[negative, "timestamp"],
        data.loc[negative, "modified_z_score"],
        color="#2ca02c",
        marker="v",
        s=25,
        zorder=4,
    )
    axis.scatter(
        data.loc[closures, "timestamp"],
        data.loc[closures, "modified_z_score"],
        color="#7f7f7f",
        marker="x",
        s=14,
        alpha=0.55,
        label="Known closure",
        zorder=3,
    )

    axis.set_title(
        "Robust Residual Scores and Detection Thresholds"
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Modified z-score")
    axis.legend(
        loc="upper left",
        ncol=2,
    )
    axis.grid(
        alpha=0.2,
    )

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(
        score_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    daily_path = (
        output / "15_daily_anomaly_counts.png"
    )

    daily_source = data.copy()
    daily_source["date"] = (
        daily_source["timestamp"].dt.floor("D")
    )

    all_dates = pd.date_range(
        daily_source["date"].min(),
        daily_source["date"].max(),
        freq="D",
    )

    positive_daily = (
        daily_source.loc[positive]
        .groupby("date")
        .size()
        .reindex(
            all_dates,
            fill_value=0,
        )
    )

    negative_daily = (
        daily_source.loc[negative]
        .groupby("date")
        .size()
        .reindex(
            all_dates,
            fill_value=0,
        )
    )

    figure, axis = plt.subplots(
        figsize=(16, 5)
    )

    axis.bar(
        all_dates,
        positive_daily,
        width=0.85,
        label="Positive candidates",
        color="#d62728",
    )
    axis.bar(
        all_dates,
        negative_daily,
        width=0.85,
        bottom=positive_daily,
        label="Negative candidates",
        color="#2ca02c",
    )

    axis.set_title(
        "Daily Actionable Anomaly Counts"
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Anomalous hours")
    axis.legend(
        loc="upper left",
    )
    axis.grid(
        axis="y",
        alpha=0.2,
    )

    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(
        daily_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    return (
        timeline_path,
        score_path,
        daily_path,
    )


def write_anomaly_report(
    detection_summary: dict[str, object],
    episode_summary: dict[str, object],
    episodes: pd.DataFrame,
    top_anomalies: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write a reproducible human-readable anomaly report."""
    if not isinstance(detection_summary, dict):
        raise AnomalyDetectionError(
            "'detection_summary' must be a dictionary."
        )

    if not isinstance(episode_summary, dict):
        raise AnomalyDetectionError(
            "'episode_summary' must be a dictionary."
        )

    if episodes.empty:
        raise AnomalyDetectionError(
            "Episode table cannot be empty."
        )

    if top_anomalies.empty:
        raise AnomalyDetectionError(
            "Top-anomaly table cannot be empty."
        )

    context_counts = episode_summary[
        "episode_context_counts"
    ]
    context_hours = episode_summary[
        "episode_context_anomaly_hours"
    ]

    episode_table = episodes.sort_values(
        [
            "anomaly_hours",
            "maximum_absolute_modified_z_score",
        ],
        ascending=[False, False],
    ).head(10)

    strongest_table = top_anomalies.head(10)

    lines = [
        "# Residual Anomaly Report",
        "",
        "## Purpose",
        "",
        (
            "This report identifies unusually large out-of-sample "
            "forecast residuals. Detected rows are candidate alerts, "
            "not automatically confirmed real-world anomalies."
        ),
        "",
        "## Detection method",
        "",
        (
            "Residuals are defined as actual demand minus forecast "
            "demand. A median-and-MAD modified z-score threshold of "
            f"{detection_summary['modified_z_threshold']} is used."
        ),
        "",
        (
            "Known service closures are preserved and scored but "
            "excluded from threshold calibration and actionable alerts."
        ),
        "",
        "## Detection summary",
        "",
        "| Measure | Value |",
        "|---|---:|",
        (
            "| Out-of-sample rows | "
            f"{detection_summary['row_count']} |"
        ),
        (
            "| Nonclosure reference rows | "
            f"{detection_summary['reference_row_count']} |"
        ),
        (
            "| Known closures | "
            f"{detection_summary['known_closure_count']} |"
        ),
        (
            "| Actionable anomaly hours | "
            f"{detection_summary['actionable_anomaly_count']} |"
        ),
        (
            "| Positive anomaly hours | "
            f"{detection_summary['positive_anomaly_count']} |"
        ),
        (
            "| Negative anomaly hours | "
            f"{detection_summary['negative_anomaly_count']} |"
        ),
        (
            "| Actionable rate | "
            f"{float(detection_summary['actionable_anomaly_rate_percent']):.2f}% |"
        ),
        "",
        "## Episode summary",
        "",
        (
            f"The {detection_summary['actionable_anomaly_count']} "
            "actionable hours form "
            f"{episode_summary['episode_count']} consecutive episodes."
        ),
        "",
        (
            "- Forecast-floor positive episodes: "
            f"{context_counts.get('forecast_floor_positive_episode', 0)} "
            "episodes covering "
            f"{context_hours.get('forecast_floor_positive_episode', 0)} "
            "hours."
        ),
        (
            "- Rain-coincident negative episodes: "
            f"{context_counts.get('rain_coincident_negative_episode', 0)} "
            "episodes covering "
            f"{context_hours.get('rain_coincident_negative_episode', 0)} "
            "hours."
        ),
        (
            "- Other residual episodes: "
            f"{context_counts.get('other_residual_episode', 0)} "
            "episodes covering "
            f"{context_hours.get('other_residual_episode', 0)} "
            "hours."
        ),
        (
            "- Share of anomalous hours occurring on the ten most "
            "concentrated dates: "
            f"{float(episode_summary['top_ten_dates_share_percent']):.2f}%."
        ),
        "",
        "## Ten largest episodes",
        "",
        (
            "| ID | Start | End | Hours | Direction | Context | "
            "Maximum score |"
        ),
        "|---:|---|---|---:|---|---|---:|",
    ]

    for _, row in episode_table.iterrows():
        lines.append(
            "| "
            f"{int(row['episode_id'])} | "
            f"{row['start_timestamp']} | "
            f"{row['end_timestamp']} | "
            f"{int(row['anomaly_hours'])} | "
            f"{row['episode_direction']} | "
            f"{row['episode_context']} | "
            f"{float(row['maximum_absolute_modified_z_score']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Ten strongest anomalous hours",
            "",
            (
                "| Rank | Timestamp | Actual | Forecast | Residual | "
                "Score | Type | Rainfall |"
            ),
            "|---:|---|---:|---:|---:|---:|---|---:|",
        ]
    )

    for _, row in strongest_table.iterrows():
        lines.append(
            "| "
            f"{int(row['anomaly_rank'])} | "
            f"{row['timestamp']} | "
            f"{float(row['actual']):.0f} | "
            f"{float(row['forecast']):.2f} | "
            f"{float(row['residual']):.2f} | "
            f"{float(row['absolute_modified_z_score']):.2f} | "
            f"{row['anomaly_type']} | "
            f"{float(row['rainfall_mm']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Rain-coincident negative episodes identify demand "
                "drops observed during rainfall. This association does "
                "not by itself prove that rainfall caused the drop."
            ),
            "",
            (
                "Forecast-floor positive episodes occur when recursive "
                "forecasts approach zero before observed demand returns "
                "to ordinary or high levels. These are important model-"
                "recovery failures rather than confirmed unusual demand."
            ),
            "",
            (
                "Other residual episodes require contextual review. "
                "They may reflect unmodeled events, changing demand, "
                "weather not represented in the forecast, or ordinary "
                "forecast error."
            ),
            "",
            "## Limitations",
            "",
            (
                "- The dataset contains no externally verified anomaly "
                "labels."
            ),
            (
                "- Statistical alerts are candidates requiring domain "
                "review."
            ),
            (
                "- The forecasting model excludes future weather because "
                "reliable future weather was not assumed available."
            ),
            (
                "- Recursive forecasting can propagate errors through "
                "later lag and rolling features."
            ),
            (
                "- The 3.5 threshold is a documented statistical rule, "
                "not a guarantee of operational importance."
            ),
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return destination