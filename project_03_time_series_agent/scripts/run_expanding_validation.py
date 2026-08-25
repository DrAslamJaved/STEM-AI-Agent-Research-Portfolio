"""Evaluate baselines across 12 expanding weekly windows."""

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.evaluation import (
    create_expanding_window_plots,
    evaluate_models_expanding_window,
    save_expanding_window_results,
    summarize_expanding_window_results,
)
from time_series_agent.preprocessing import preprocess_time_series


CONFIG_PATH = "configs/default.yaml"

NUMBER_OF_FOLDS = 12
HORIZON = 168
STEP = 168
MASE_PERIOD = 24

FOLD_RESULTS_PATH = (
    "reports/metrics/baseline_expanding_fold_results.csv"
)
SUMMARY_PATH = (
    "reports/metrics/baseline_expanding_summary.csv"
)
FIGURE_DIRECTORY = "reports/figures"


def main() -> None:
    """Run leakage-safe expanding-window baseline evaluation."""
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

    if initial_train_size <= 168:
        raise ValueError(
            "Insufficient initial training data for validation."
        )

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

    fold_results = evaluate_models_expanding_window(
        series=complete_series,
        model_factories=factories,
        initial_train_size=initial_train_size,
        horizon=HORIZON,
        step=STEP,
        mase_period=MASE_PERIOD,
    )

    summary = summarize_expanding_window_results(
        fold_results
    )

    save_expanding_window_results(
        fold_results=fold_results,
        summary=summary,
        fold_results_path=FOLD_RESULTS_PATH,
        summary_path=SUMMARY_PATH,
    )

    figure_paths = create_expanding_window_plots(
        fold_results=fold_results,
        summary=summary,
        output_directory=FIGURE_DIRECTORY,
    )

    print("Expanding-window configuration:")
    print(f"- Folds: {NUMBER_OF_FOLDS}")
    print(f"- Initial training rows: {initial_train_size}")
    print(f"- Horizon per fold: {HORIZON}")
    print(f"- Step: {STEP}")
    print()
    print("Aggregate results:")
    print(summary.to_string(index=False))
    print()
    print("Generated figures:")

    for figure_path in figure_paths:
        print(f"- {figure_path}")


if __name__ == "__main__":
    main()