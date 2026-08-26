"""Compare Holt-Winters with baselines across 12 folds."""

import json
from pathlib import Path

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.classical import HoltWintersForecaster
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
    "reports/metrics/classical_expanding_fold_results.csv"
)
SUMMARY_PATH = (
    "reports/metrics/classical_expanding_summary.csv"
)
COMPARISON_PATH = (
    "reports/metrics/holt_winters_vs_baseline.json"
)
FIGURE_DIRECTORY = "reports/figures"


def main() -> None:
    """Run leakage-safe Holt-Winters model comparison."""
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
        "holt_winters_24": (
            lambda: HoltWintersForecaster(
                seasonal_period=24,
                damped_trend=True,
                clip_nonnegative=True,
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
        fold_filename=(
            "09_holt_winters_mae_by_fold.png"
        ),
        summary_filename=(
            "10_holt_winters_mean_mae.png"
        ),
        title_prefix="Baseline and Holt-Winters",
    )

    baseline_row = summary.loc[
        summary["model"].eq("seasonal_naive_168")
    ].iloc[0]
    candidate_row = summary.loc[
        summary["model"].eq("holt_winters_24")
    ].iloc[0]

    baseline_mae = float(baseline_row["mean_mae"])
    candidate_mae = float(candidate_row["mean_mae"])

    improvement_percentage = float(
        100
        * (baseline_mae - candidate_mae)
        / baseline_mae
    )

    comparison = {
        "evaluation_folds": NUMBER_OF_FOLDS,
        "forecast_horizon_per_fold": HORIZON,
        "baseline_model": "seasonal_naive_168",
        "baseline_mean_mae": baseline_mae,
        "candidate_model": "holt_winters_24",
        "candidate_mean_mae": candidate_mae,
        "candidate_mae_degradation_percentage": float(
            100
            * (candidate_mae - baseline_mae)
            / baseline_mae
        ),
        "candidate_mae_improvement_percentage": (
            improvement_percentage
        ),
        "candidate_beats_baseline": bool(
            candidate_mae < baseline_mae
        ),
        "candidate_mae_rank": int(
            candidate_row["mae_rank"]
        ),
        "candidate_mae_fold_wins": int(
            candidate_row["mae_fold_wins"]
        ),
        "candidate_total_raw_negative_forecasts": int(
            candidate_row[
                "total_raw_negative_forecasts"
            ]
        ),
    }

    comparison_output = Path(COMPARISON_PATH)
    comparison_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    comparison_output.write_text(
        json.dumps(
            comparison,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Classical-model expanding-window results:")
    print(summary.to_string(index=False))
    print()
    print("Holt-Winters comparison:")
    print(json.dumps(comparison, indent=2))
    print()
    print("Generated figures:")

    for figure_path in figure_paths:
        print(f"- {figure_path}")


if __name__ == "__main__":
    main()