"""Run reproducible exploratory time-series analysis."""

import json

from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.exploration import (
    compute_exploration_summary,
    create_exploration_figures,
    save_exploration_summary,
)
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"
SUMMARY_PATH = "reports/metrics/exploration_summary.json"
FIGURE_DIRECTORY = "reports/figures"


def main() -> None:
    """Run the complete exploration workflow."""
    config = load_data_config(CONFIG_PATH)
    loaded_data = load_time_series_csv(config)

    processed_data, _ = preprocess_time_series(
        data=loaded_data,
        config=config,
        closure_column="Functioning Day",
        closure_value="No",
    )

    summary = compute_exploration_summary(
        data=processed_data,
        timestamp_column=config.timestamp_column,
        target_column=config.target_column,
    )

    figure_paths = create_exploration_figures(
        data=processed_data,
        timestamp_column=config.timestamp_column,
        target_column=config.target_column,
        output_directory=FIGURE_DIRECTORY,
        seasonal_period=24,
        maximum_lag=72,
    )

    save_exploration_summary(
        summary=summary,
        output_path=SUMMARY_PATH,
    )

    print(json.dumps(summary.to_dict(), indent=2))
    print("Generated figures:")

    for figure_path in figure_paths:
        print(f"- {figure_path}")


if __name__ == "__main__":
    main()