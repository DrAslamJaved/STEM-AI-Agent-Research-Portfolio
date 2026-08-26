"""Generate anomaly figures and the human-readable report."""

import json
from pathlib import Path

import pandas as pd

from time_series_agent.anomaly_reporting import (
    create_anomaly_report_figures,
    enrich_anomaly_context,
    write_anomaly_report,
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
DETECTION_SUMMARY_PATH = (
    "reports/metrics/anomaly_detection_summary.json"
)
EPISODE_SUMMARY_PATH = (
    "reports/metrics/anomaly_episode_summary.json"
)
FIGURE_DIRECTORY = "reports/figures"
REPORT_PATH = "reports/anomaly_report.md"


def load_json(path: str) -> dict[str, object]:
    """Load a structured JSON report."""
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    """Create the final anomaly-reporting artifacts."""
    config = load_data_config(CONFIG_PATH)
    source_data = load_time_series_csv(config)

    labels = pd.read_csv(
        LABELS_PATH,
        parse_dates=["timestamp"],
    )
    episodes = pd.read_csv(
        EPISODES_PATH,
        parse_dates=[
            "start_timestamp",
            "end_timestamp",
            "peak_timestamp",
        ],
    )
    top_anomalies = pd.read_csv(
        TOP_ANOMALIES_PATH,
        parse_dates=["timestamp"],
    )

    detection_summary = load_json(
        DETECTION_SUMMARY_PATH
    )
    episode_summary = load_json(
        EPISODE_SUMMARY_PATH
    )

    contextual = enrich_anomaly_context(
        labels=labels,
        source_data=source_data,
        timestamp_column=config.timestamp_column,
    )

    figure_paths = create_anomaly_report_figures(
        contextual_labels=contextual,
        output_directory=FIGURE_DIRECTORY,
        threshold=float(
            detection_summary[
                "modified_z_threshold"
            ]
        ),
    )

    report_path = write_anomaly_report(
        detection_summary=detection_summary,
        episode_summary=episode_summary,
        episodes=episodes,
        top_anomalies=top_anomalies,
        output_path=REPORT_PATH,
    )

    print("Generated anomaly figures:")

    for figure_path in figure_paths:
        print(f"- {figure_path}")

    print()
    print(f"Anomaly report: {report_path}")


if __name__ == "__main__":
    main()