"""Create contextual anomaly episodes and ranked alerts."""

import json
from pathlib import Path

import pandas as pd

from time_series_agent.anomaly_reporting import (
    build_anomaly_episodes,
    enrich_anomaly_context,
    select_top_actionable_anomalies,
)
from time_series_agent.config import load_data_config
from time_series_agent.data_loader import (
    load_time_series_csv,
)


CONFIG_PATH = "configs/default.yaml"

LABELS_PATH = (
    "reports/metrics/"
    "gradient_boosting_anomaly_labels.csv"
)
EPISODES_PATH = (
    "reports/metrics/anomaly_episodes.csv"
)
TOP_ANOMALIES_PATH = (
    "reports/metrics/top_actionable_anomalies.csv"
)
SUMMARY_PATH = (
    "reports/metrics/anomaly_episode_summary.json"
)

MAXIMUM_GAP_HOURS = 1
FORECAST_FLOOR_THRESHOLD = 50.0
TOP_LIMIT = 20


def main() -> None:
    """Build anomaly episodes with observed context."""
    config = load_data_config(CONFIG_PATH)
    source_data = load_time_series_csv(config)

    labels = pd.read_csv(
        LABELS_PATH,
        parse_dates=["timestamp"],
    )

    contextual = enrich_anomaly_context(
        labels=labels,
        source_data=source_data,
        timestamp_column=config.timestamp_column,
    )

    episodes = build_anomaly_episodes(
        contextual_labels=contextual,
        maximum_gap_hours=MAXIMUM_GAP_HOURS,
        forecast_floor_threshold=(
            FORECAST_FLOOR_THRESHOLD
        ),
    )

    top_anomalies = select_top_actionable_anomalies(
        contextual_labels=contextual,
        limit=TOP_LIMIT,
    )

    episodes_output = Path(EPISODES_PATH)
    top_output = Path(TOP_ANOMALIES_PATH)

    episodes_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    top_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    episodes.to_csv(
        episodes_output,
        index=False,
    )
    top_anomalies.to_csv(
        top_output,
        index=False,
    )

    actionable = contextual.loc[
        contextual["is_actionable_anomaly"]
    ].copy()
    actionable["date"] = actionable[
        "timestamp"
    ].dt.date

    top_ten_date_hours = int(
        actionable.groupby("date")
        .size()
        .nlargest(10)
        .sum()
    )

    context_counts = {
        str(key): int(value)
        for key, value in episodes[
            "episode_context"
        ].value_counts().items()
    }

    context_hours = {
        str(key): int(value)
        for key, value in episodes.groupby(
            "episode_context"
        )["anomaly_hours"].sum().items()
    }

    summary = {
        "maximum_gap_hours": MAXIMUM_GAP_HOURS,
        "forecast_floor_threshold": (
            FORECAST_FLOOR_THRESHOLD
        ),
        "actionable_anomaly_hours": int(
            len(actionable)
        ),
        "episode_count": int(len(episodes)),
        "episode_context_counts": context_counts,
        "episode_context_anomaly_hours": (
            context_hours
        ),
        "longest_episode_hours": int(
            episodes["duration_hours"].max()
        ),
        "largest_episode_anomaly_hours": int(
            episodes["anomaly_hours"].max()
        ),
        "top_ten_dates_anomaly_hours": (
            top_ten_date_hours
        ),
        "top_ten_dates_share_percent": float(
            100
            * top_ten_date_hours
            / len(actionable)
        ),
        "strongest_episode_id": int(
            episodes.loc[
                episodes[
                    "maximum_absolute_modified_z_score"
                ].idxmax(),
                "episode_id",
            ]
        ),
        "maximum_episode_score": float(
            episodes[
                "maximum_absolute_modified_z_score"
            ].max()
        ),
    }

    summary_output = Path(SUMMARY_PATH)
    summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Anomaly-episode summary:")
    print(json.dumps(summary, indent=2))
    print()
    print("Largest episodes:")
    print(
        episodes.sort_values(
            [
                "anomaly_hours",
                "maximum_absolute_modified_z_score",
            ],
            ascending=[False, False],
        )
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()