"""Compare Gradient Boosting with earlier forecasting models."""

import json
from pathlib import Path

from time_series_agent.baselines import (
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from time_series_agent.classical import (
    HoltWintersForecaster,
)
from time_series_agent.config import load_data_config
from time_series_agent.data_loader import (
    load_time_series_csv,
)
from time_series_agent.evaluation import (
    create_expanding_window_plots,
    evaluate_models_expanding_window,
    save_expanding_window_results,
    summarize_expanding_window_results,
)
from time_series_agent.machine_learning import (
    RecursiveGradientBoostingForecaster,
)
from time_series_agent.preprocessing import (
    preprocess_time_series,
)


CONFIG_PATH = "configs/default.yaml"

NUMBER_OF_FOLDS = 12
HORIZON = 168
STEP = 168
MASE_PERIOD = 24

FOLD_RESULTS_PATH = (
    "reports/metrics/ml_expanding_fold_results.csv"
)
SUMMARY_PATH = (
    "reports/metrics/ml_expanding_summary.csv"
)
COMPARISON_PATH = (
    "reports/metrics/"
    "gradient_boosting_vs_baseline.json"
)
FIGURE_DIRECTORY = "reports/figures"


def main() -> None:
    """Run leakage-safe machine-learning comparison."""
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
        "mean": (
            lambda: MeanForecaster(
                frequency="h",
            )
        ),
        "naive": (
            lambda: NaiveForecaster(
                frequency="h",
            )
        ),
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
        "gradient_boosting_recursive": (
            lambda: RecursiveGradientBoostingForecaster(
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
            "11_machine_learning_mae_by_fold.png"
        ),
        summary_filename=(
            "12_machine_learning_mean_mae.png"
        ),
        title_prefix=(
            "Baseline, Classical, and Machine Learning"
        ),
    )

    baseline_row = summary.loc[
        summary["model"].eq(
            "seasonal_naive_168"
        )
    ].iloc[0]

    classical_row = summary.loc[
        summary["model"].eq(
            "holt_winters_24"
        )
    ].iloc[0]

    candidate_row = summary.loc[
        summary["model"].eq(
            "gradient_boosting_recursive"
        )
    ].iloc[0]

    best_row = summary.sort_values(
        by=["mean_mae", "model"],
        ascending=[True, True],
    ).iloc[0]

    baseline_mae = float(
        baseline_row["mean_mae"]
    )
    classical_mae = float(
        classical_row["mean_mae"]
    )
    candidate_mae = float(
        candidate_row["mean_mae"]
    )

    improvement_percentage = float(
        100
        * (baseline_mae - candidate_mae)
        / baseline_mae
    )

    degradation_percentage = float(
        100
        * (candidate_mae - baseline_mae)
        / baseline_mae
    )

    comparison = {
        "evaluation_folds": NUMBER_OF_FOLDS,
        "forecast_horizon_per_fold": HORIZON,
        "total_candidate_test_predictions": (
            NUMBER_OF_FOLDS * HORIZON
        ),
        "baseline_model": "seasonal_naive_168",
        "baseline_mean_mae": baseline_mae,
        "classical_model": "holt_winters_24",
        "classical_mean_mae": classical_mae,
        "candidate_model": (
            "gradient_boosting_recursive"
        ),
        "candidate_mean_mae": candidate_mae,
        "candidate_mae_difference_from_baseline": (
            candidate_mae - baseline_mae
        ),
        "candidate_mae_improvement_percentage": (
            improvement_percentage
        ),
        "candidate_mae_degradation_percentage": (
            degradation_percentage
        ),
        "candidate_beats_baseline": bool(
            candidate_mae < baseline_mae
        ),
        "candidate_beats_holt_winters": bool(
            candidate_mae < classical_mae
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
        "best_model": str(
            best_row["model"]
        ),
        "best_mean_mae": float(
            best_row["mean_mae"]
        ),
        "candidate_is_best_model": bool(
            best_row["model"]
            == "gradient_boosting_recursive"
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

    print("Machine-learning expanding-window results:")
    print(summary.to_string(index=False))
    print()
    print("Gradient Boosting comparison:")
    print(json.dumps(comparison, indent=2))
    print()
    print("Generated figures:")

    for figure_path in figure_paths:
        print(f"- {figure_path}")


if __name__ == "__main__":
    main()