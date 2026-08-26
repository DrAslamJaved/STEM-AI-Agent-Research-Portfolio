"""Fit recursive Gradient Boosting and preview 24 hours."""

import json
from pathlib import Path

from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.machine_learning import (
    RecursiveGradientBoostingForecaster,
)
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"
FORECAST_PATH = (
    "reports/metrics/"
    "gradient_boosting_next_24_hours.csv"
)
DIAGNOSTICS_PATH = (
    "reports/metrics/"
    "gradient_boosting_diagnostics.json"
)


def main() -> None:
    """Fit the model and save forecast and diagnostics."""
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

    model = RecursiveGradientBoostingForecaster(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        clip_nonnegative=True,
        frequency="h",
    )

    model.fit(training_series)
    forecast = model.predict(24)
    forecast.name = "gradient_boosting_recursive"

    forecast.to_csv(
        FORECAST_PATH,
        index=True,
        index_label="timestamp",
        encoding="utf-8",
    )

    diagnostics_payload = {
        "fit": model.diagnostics().to_dict(),
        "forecast": {
            "horizon": len(forecast),
            "recursive_strategy": True,
            "nonnegative_constraint": True,
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

    output = Path(DIAGNOSTICS_PATH)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.write_text(
        json.dumps(
            diagnostics_payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(forecast)
    print()
    print(
        json.dumps(
            diagnostics_payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()