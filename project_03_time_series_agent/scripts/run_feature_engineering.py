"""Construct and summarize leakage-safe forecasting features."""

import json

from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.features import (
    build_lag_feature_set,
    create_feature_summary,
    save_feature_summary,
)
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"
SUMMARY_PATH = (
    "reports/metrics/feature_engineering_summary.json"
)


def main() -> None:
    """Build leakage-safe features from the real time series."""
    config = load_data_config(CONFIG_PATH)
    loaded_data = load_time_series_csv(config)

    processed_data, _ = preprocess_time_series(
        data=loaded_data,
        config=config,
        closure_column="Functioning Day",
        closure_value="No",
    )

    feature_set = build_lag_feature_set(
        data=processed_data,
        timestamp_column=config.timestamp_column,
        target_column=config.target_column,
        lags=(1, 24, 168),
        rolling_windows=(24, 168),
    )

    summary = create_feature_summary(
        feature_set=feature_set,
        lags=(1, 24, 168),
        rolling_windows=(24, 168),
    )

    save_feature_summary(
        summary,
        SUMMARY_PATH,
    )

    print(json.dumps(summary.to_dict(), indent=2))
    print()
    print("First three feature rows:")
    print(feature_set.features.head(3).to_string())
    print()
    print("Feature engineering does not use future weather data.")


if __name__ == "__main__":
    main()