"""Collect out-of-sample Gradient Boosting residuals."""

import json
from pathlib import Path

from time_series_agent.config import load_data_config
from time_series_agent.data_loader import (
    load_time_series_csv,
)
from time_series_agent.exceptions import EvaluationError
from time_series_agent.machine_learning import (
    RecursiveGradientBoostingForecaster,
)
from time_series_agent.preprocessing import (
    preprocess_time_series,
)
from time_series_agent.residuals import (
    collect_expanding_window_residuals,
    save_expanding_window_residuals,
)


CONFIG_PATH = "configs/default.yaml"

NUMBER_OF_FOLDS = 12
HORIZON = 168
STEP = 168

RESIDUAL_PATH = (
    "reports/metrics/"
    "gradient_boosting_oos_residuals.csv"
)
SUMMARY_PATH = (
    "reports/metrics/"
    "oos_residual_collection_summary.json"
)


def model_factory() -> RecursiveGradientBoostingForecaster:
    """Create the selected forecasting model."""
    return RecursiveGradientBoostingForecaster(
        lags=(1, 24, 168),
        rolling_windows=(24, 168),
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        clip_nonnegative=True,
        frequency="h",
    )


def main() -> None:
    """Generate leakage-safe residual evidence."""
    config = load_data_config(CONFIG_PATH)
    loaded_data = load_time_series_csv(config)

    processed_data, _ = preprocess_time_series(
        data=loaded_data,
        config=config,
        closure_column="Functioning Day",
        closure_value="No",
    )

    complete_series = processed_data.set_index(
        config.timestamp_column
    )[config.target_column]

    initial_train_size = (
        len(complete_series)
        - NUMBER_OF_FOLDS * HORIZON
    )

    residuals = collect_expanding_window_residuals(
        series=complete_series,
        model_name="gradient_boosting_recursive",
        model_factory=model_factory,
        initial_train_size=initial_train_size,
        horizon=HORIZON,
        step=STEP,
    )

    closure_lookup = processed_data.set_index(
        config.timestamp_column
    )["is_known_closure"]

    closure_flags = residuals["timestamp"].map(
        closure_lookup
    )

    if closure_flags.isna().any():
        raise EvaluationError(
            "Closure status is unavailable for one or "
            "more residual timestamps."
        )

    residuals["is_known_closure"] = (
        closure_flags.astype(bool)
    )

    residual_output = (
        save_expanding_window_residuals(
            residuals=residuals,
            output_path=RESIDUAL_PATH,
        )
    )

    fold_negative_counts = (
        residuals[
            [
                "fold",
                "fold_raw_negative_forecast_count",
            ]
        ]
        .drop_duplicates(subset="fold")
    )

    summary = {
        "model": "gradient_boosting_recursive",
        "fold_count": int(
            residuals["fold"].nunique()
        ),
        "forecast_horizon_per_fold": HORIZON,
        "row_count": int(len(residuals)),
        "unique_timestamp_count": int(
            residuals["timestamp"].nunique()
        ),
        "start_timestamp": str(
            residuals["timestamp"].min()
        ),
        "end_timestamp": str(
            residuals["timestamp"].max()
        ),
        "actual_minimum": float(
            residuals["actual"].min()
        ),
        "actual_maximum": float(
            residuals["actual"].max()
        ),
        "forecast_minimum": float(
            residuals["forecast"].min()
        ),
        "forecast_maximum": float(
            residuals["forecast"].max()
        ),
        "residual_mean": float(
            residuals["residual"].mean()
        ),
        "residual_median": float(
            residuals["residual"].median()
        ),
        "residual_standard_deviation": float(
            residuals["residual"].std(ddof=1)
        ),
        "mean_absolute_residual": float(
            residuals["absolute_residual"].mean()
        ),
        "median_absolute_residual": float(
            residuals["absolute_residual"].median()
        ),
        "maximum_absolute_residual": float(
            residuals["absolute_residual"].max()
        ),
        "known_closure_count": int(
            residuals["is_known_closure"].sum()
        ),
        "zero_actual_count": int(
            residuals["actual"].eq(0).sum()
        ),
        "constrained_negative_forecast_count": int(
            residuals["forecast"].lt(0).sum()
        ),
        "total_raw_negative_forecasts": int(
            fold_negative_counts[
                "fold_raw_negative_forecast_count"
            ].sum()
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

    print("Out-of-sample residual collection:")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Residual table: {residual_output}")


if __name__ == "__main__":
    main()