"""Detect robust anomalies in out-of-sample residuals."""

import json

import pandas as pd

from time_series_agent.anomalies import (
    detect_residual_anomalies,
    save_anomaly_results,
)


RESIDUAL_PATH = (
    "reports/metrics/"
    "gradient_boosting_oos_residuals.csv"
)
LABELS_PATH = (
    "reports/metrics/"
    "gradient_boosting_anomaly_labels.csv"
)
SUMMARY_PATH = (
    "reports/metrics/"
    "anomaly_detection_summary.json"
)

MODIFIED_Z_THRESHOLD = 3.5
MINIMUM_REFERENCE_ROWS = 20


def main() -> None:
    """Run robust residual anomaly detection."""
    residuals = pd.read_csv(
        RESIDUAL_PATH,
        parse_dates=["timestamp"],
    )

    labeled, summary = detect_residual_anomalies(
        residuals=residuals,
        threshold=MODIFIED_Z_THRESHOLD,
        minimum_reference_rows=(
            MINIMUM_REFERENCE_ROWS
        ),
    )

    labels_output, summary_output = (
        save_anomaly_results(
            labeled_anomalies=labeled,
            summary=summary,
            labels_path=LABELS_PATH,
            summary_path=SUMMARY_PATH,
        )
    )

    top_anomalies = (
        labeled.loc[
            labeled["is_actionable_anomaly"],
            [
                "timestamp",
                "actual",
                "forecast",
                "residual",
                "modified_z_score",
                "anomaly_type",
                "anomaly_severity",
            ],
        ]
        .sort_values(
            "absolute_modified_z_score",
            ascending=False,
        )
        .head(10)
    )

    print("Anomaly-detection summary:")
    print(
        json.dumps(
            summary.to_dict(),
            indent=2,
        )
    )
    print()
    print("Top actionable anomalies:")
    print(
        top_anomalies.to_string(
            index=False
        )
    )
    print()
    print(f"Labels: {labels_output}")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()