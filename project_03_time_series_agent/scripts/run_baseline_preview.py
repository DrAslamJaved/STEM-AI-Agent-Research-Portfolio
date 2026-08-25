"""Generate unevaluated next-period baseline forecast previews."""

import pandas as pd

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"
OUTPUT_PATH = "reports/metrics/baseline_next_24_hours.csv"


def main() -> None:
    """Fit four baselines and forecast the next 24 hours."""
    config = load_data_config(CONFIG_PATH)
    loaded_data = load_time_series_csv(config)

    processed_data, _ = preprocess_time_series(
        data=loaded_data,
        config=config,
        closure_column="Functioning Day",
        closure_value="No",
    )

    training_series = processed_data.set_index(
        config.timestamp_column
    )[config.target_column]

    models = {
        "mean": MeanForecaster(frequency="h"),
        "naive": NaiveForecaster(frequency="h"),
        "seasonal_naive_24": SeasonalNaiveForecaster(
            seasonal_period=24,
            frequency="h",
        ),
        "seasonal_naive_168": SeasonalNaiveForecaster(
            seasonal_period=168,
            frequency="h",
        ),
    }

    forecasts: list[pd.Series] = []

    for model_name, model in models.items():
        forecast = model.fit(training_series).predict(24)
        forecast.name = model_name
        forecasts.append(forecast)

    forecast_table = pd.concat(forecasts, axis=1)
    forecast_table.index.name = "timestamp"

    forecast_table.to_csv(
        OUTPUT_PATH,
        encoding="utf-8",
    )

    print(forecast_table)
    print()
    print("Important: these forecasts have not yet been evaluated.")
    print(f"Saved forecast preview to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()