"""Generate the final forecasting-model recommendation."""

import json
from pathlib import Path

import pandas as pd

from time_series_agent.exceptions import (
    RecommendationError,
)
from time_series_agent.recommendation import (
    recommend_forecasting_model,
    save_model_recommendation,
    write_model_recommendation_report,
)


SUMMARY_PATH = (
    "reports/metrics/ml_expanding_summary.csv"
)
FOLD_RESULTS_PATH = (
    "reports/metrics/ml_expanding_fold_results.csv"
)
ANOMALY_SUMMARY_PATH = (
    "reports/metrics/anomaly_detection_summary.json"
)
EPISODE_SUMMARY_PATH = (
    "reports/metrics/anomaly_episode_summary.json"
)

RECOMMENDATION_JSON_PATH = (
    "reports/metrics/model_recommendation.json"
)
RECOMMENDATION_REPORT_PATH = (
    "reports/model_recommendation.md"
)

BENCHMARK_MODEL = "seasonal_naive_168"
MINIMUM_MAE_IMPROVEMENT_PERCENTAGE = 2.0


def load_json(path: str) -> dict[str, object]:
    """Load one structured JSON artifact."""
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    """Run the model-selection policy."""
    model_summary = pd.read_csv(
        SUMMARY_PATH
    )
    fold_results = pd.read_csv(
        FOLD_RESULTS_PATH
    )

    test_totals = (
        fold_results.groupby("model")[
            "test_rows"
        ]
        .sum()
    )

    if test_totals.nunique() != 1:
        raise RecommendationError(
            "Models do not have equal numbers of "
            "out-of-sample predictions."
        )

    total_predictions_per_model = int(
        test_totals.iloc[0]
    )

    recommendation = recommend_forecasting_model(
        summary=model_summary,
        benchmark_model=BENCHMARK_MODEL,
        minimum_mae_improvement_percentage=(
            MINIMUM_MAE_IMPROVEMENT_PERCENTAGE
        ),
        total_predictions_per_model=(
            total_predictions_per_model
        ),
    )

    recommendation_json = (
        save_model_recommendation(
            recommendation=recommendation,
            output_path=(
                RECOMMENDATION_JSON_PATH
            ),
        )
    )

    anomaly_summary = load_json(
        ANOMALY_SUMMARY_PATH
    )
    episode_summary = load_json(
        EPISODE_SUMMARY_PATH
    )

    recommendation_report = (
        write_model_recommendation_report(
            recommendation=recommendation,
            model_summary=model_summary,
            anomaly_summary=anomaly_summary,
            episode_summary=episode_summary,
            output_path=(
                RECOMMENDATION_REPORT_PATH
            ),
        )
    )

    print("Model recommendation:")
    print(
        json.dumps(
            recommendation.to_dict(),
            indent=2,
        )
    )
    print()
    print(f"JSON decision: {recommendation_json}")
    print(
        f"Human-readable report: "
        f"{recommendation_report}"
    )


if __name__ == "__main__":
    main()