"""Evaluate baseline models on the final one-week holdout."""

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.evaluation import (
    create_holdout_comparison_plot,
    evaluate_models_on_holdout,
    save_holdout_results,
)
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"
HOLDOUT_HORIZON = 168
MASE_PERIOD = 24

METRICS_PATH = "reports/metrics/baseline_holdout_metrics.csv"
FORECASTS_PATH = "reports/metrics/baseline_holdout_forecasts.csv"
FIGURE_PATH = "reports/figures/06_baseline_holdout_comparison.png"


def main() -> None:
    """Run leakage-safe one-week baseline evaluation."""
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

    factories = {
        "mean": lambda: MeanForecaster(frequency="h"),
        "naive": lambda: NaiveForecaster(frequency="h"),
        "seasonal_naive_24": (
            lambda: SeasonalNaiveForecaster(
                seasonal_period=24,
                frequency="h",
            )
        ),
        "seasonal_naive_168": (
            lambda: SeasonalNaiveForecaster(
                seasonal_period=168,
                frequency="h",
            )
        ),
    }

    metrics, forecasts = evaluate_models_on_holdout(
        series=complete_series,
        model_factories=factories,
        horizon=HOLDOUT_HORIZON,
        mase_period=MASE_PERIOD,
    )

    save_holdout_results(
        metrics=metrics,
        forecasts=forecasts,
        metrics_path=METRICS_PATH,
        forecasts_path=FORECASTS_PATH,
    )
    create_holdout_comparison_plot(
        forecasts=forecasts,
        output_path=FIGURE_PATH,
    )

    print("Chronological holdout results:")
    print(metrics.to_string(index=False))
    print()
    print(
        "The lowest-MAE model is provisional because this is "
        "only one holdout period."
    )
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Forecasts saved to: {FORECASTS_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()