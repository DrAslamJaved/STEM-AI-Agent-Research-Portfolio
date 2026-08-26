"""Fit Holt-Winters and preview the next 24 hourly forecasts."""

import json
from pathlib import Path

from time_series_agent.classical import HoltWintersForecaster
from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"
FORECAST_PATH = (
    "reports/metrics/holt_winters_next_24_hours.csv"
)
DIAGNOSTICS_PATH = (
    "reports/metrics/holt_winters_diagnostics.json"
)


def main() -> None:
    """Fit the daily seasonal model and save its preview."""
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

    model = HoltWintersForecaster(
        seasonal_period=24,
        damped_trend=True,
        frequency="h",
    )
    model.fit(training_series)

    forecast = model.predict(24)
    forecast.name = "holt_winters_24"

    forecast.to_csv(
        FORECAST_PATH,
        index=True,
        index_label="timestamp",
        encoding="utf-8",
    )

    fit_diagnostics = model.diagnostics()

    diagnostics_payload = {
        "fit": fit_diagnostics.to_dict(),
        "forecast": {
            "horizon": len(forecast),
            "nonnegative_constraint": (
                model.clip_nonnegative
            ),
            "raw_negative_forecast_count": (
                model.last_raw_negative_forecast_count()
            ),
            "constrained_minimum": float(
                forecast.min()
            ),
            "constrained_maximum": float(
                forecast.max()
            ),
        },
    }

    diagnostics_output = Path(DIAGNOSTICS_PATH)
    diagnostics_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    diagnostics_output.write_text(
        json.dumps(
            diagnostics_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(forecast)
    print()
    print("Holt-Winters diagnostics:")
    print(
        json.dumps(
            diagnostics_payload,
            indent=2,
        )
    )
    diagnostics_output.write_text(
        json.dumps(
            diagnostics.to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(forecast)
    print()
    print("Holt-Winters diagnostics:")
    print(
        json.dumps(
            diagnostics.to_dict(),
            indent=2,
        )
    )
    print()
    print(
        "Important: the next-24-hour preview is not "
        "an accuracy evaluation."
    )


if __name__ == "__main__":
    main()