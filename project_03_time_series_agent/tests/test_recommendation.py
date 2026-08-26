"""Tests for evidence-based model recommendation."""

import json

import pandas as pd
import pytest

from time_series_agent.exceptions import (
    RecommendationError,
)
from time_series_agent.recommendation import (
    recommend_forecasting_model,
    save_model_recommendation,
    write_model_recommendation_report,
)


def make_summary() -> pd.DataFrame:
    """Create representative model-comparison evidence."""
    return pd.DataFrame(
        {
            "model": [
                "gradient_boosting_recursive",
                "seasonal_naive_168",
                "seasonal_naive_24",
            ],
            "folds": [12, 12, 12],
            "mean_mae": [
                404.72,
                420.56,
                444.04,
            ],
            "std_mae": [
                230.61,
                223.24,
                270.58,
            ],
            "mean_rmse": [
                544.54,
                618.48,
                621.37,
            ],
            "mean_smape": [
                71.57,
                71.80,
                80.87,
            ],
            "mean_mase": [
                1.508,
                1.573,
                1.647,
            ],
            "mae_fold_wins": [3, 4, 3],
            "mae_rank": [1, 2, 3],
            "total_raw_negative_forecasts": [
                199,
                0,
                0,
            ],
        }
    )


def test_gradient_boosting_passes_selection_gates() -> None:
    """A meaningful MAE and RMSE improvement should win."""
    result = recommend_forecasting_model(
        make_summary(),
        minimum_mae_improvement_percentage=2.0,
        total_predictions_per_model=2016,
    )

    assert (
        result.selected_model
        == "gradient_boosting_recursive"
    )
    assert (
        result.fallback_model
        == "seasonal_naive_168"
    )
    assert result.selection_status == (
        "candidate_selected"
    )
    assert result.candidate_meets_mae_improvement_rule
    assert result.candidate_meets_rmse_rule
    assert result.observed_mae_improvement_percentage > 3
    assert result.candidate_raw_negative_rate_percentage > 9
    assert len(result.warnings) == 3


def test_small_improvement_retains_benchmark() -> None:
    """A negligible MAE gain should not justify complexity."""
    summary = make_summary()
    summary.loc[
        summary["model"].eq(
            "gradient_boosting_recursive"
        ),
        "mean_mae",
    ] = 415.0

    result = recommend_forecasting_model(
        summary,
        minimum_mae_improvement_percentage=2.0,
    )

    assert (
        result.selected_model
        == "seasonal_naive_168"
    )
    assert result.selection_status == (
        "benchmark_retained"
    )
    assert not result.candidate_meets_mae_improvement_rule


def test_worse_rmse_retains_benchmark() -> None:
    """Better MAE cannot compensate for worse RMSE."""
    summary = make_summary()
    summary.loc[
        summary["model"].eq(
            "gradient_boosting_recursive"
        ),
        "mean_rmse",
    ] = 700.0

    result = recommend_forecasting_model(summary)

    assert (
        result.selected_model
        == "seasonal_naive_168"
    )
    assert not result.candidate_meets_rmse_rule


def test_benchmark_can_already_be_best() -> None:
    """The benchmark should remain selected when ranked first."""
    summary = make_summary()
    summary.loc[
        summary["model"].eq(
            "gradient_boosting_recursive"
        ),
        "mean_mae",
    ] = 430.0

    result = recommend_forecasting_model(summary)

    assert (
        result.selected_model
        == "seasonal_naive_168"
    )
    assert result.selection_status == (
        "benchmark_already_best"
    )
    assert (
        result.fallback_model
        == "gradient_boosting_recursive"
    )


def test_missing_summary_column_is_rejected() -> None:
    """Incomplete comparison evidence must be rejected."""
    bad_summary = make_summary().drop(
        columns="mean_rmse"
    )

    with pytest.raises(
        RecommendationError,
        match="missing columns",
    ):
        recommend_forecasting_model(bad_summary)


def test_duplicate_model_name_is_rejected() -> None:
    """Every model must appear exactly once."""
    bad_summary = make_summary()
    bad_summary.loc[2, "model"] = (
        "seasonal_naive_168"
    )

    with pytest.raises(
        RecommendationError,
        match="unique",
    ):
        recommend_forecasting_model(bad_summary)


def test_unequal_fold_counts_are_rejected() -> None:
    """Models must be compared on identical fold counts."""
    bad_summary = make_summary()
    bad_summary.loc[0, "folds"] = 11

    with pytest.raises(
        RecommendationError,
        match="same number of folds",
    ):
        recommend_forecasting_model(bad_summary)


def test_invalid_prediction_count_is_rejected() -> None:
    """Raw-negative rates need a valid denominator."""
    with pytest.raises(
        RecommendationError,
        match="positive integer",
    ):
        recommend_forecasting_model(
            make_summary(),
            total_predictions_per_model=0,
        )


def test_recommendation_can_be_saved(
    tmp_path,
) -> None:
    """The model decision should be reproducible JSON."""
    recommendation = recommend_forecasting_model(
        make_summary()
    )

    output_path = (
        tmp_path / "nested" / "recommendation.json"
    )

    saved_path = save_model_recommendation(
        recommendation,
        output_path,
    )

    loaded = json.loads(
        saved_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_path.exists()
    assert (
        loaded["selected_model"]
        == "gradient_boosting_recursive"
    )
    assert (
        loaded["fallback_model"]
        == "seasonal_naive_168"
    )
    assert loaded["decision_checks"][
        "candidate_meets_rmse_rule"
    ]


def test_recommendation_report_can_be_written(
    tmp_path,
) -> None:
    """Human-readable report should explain the decision."""
    summary = make_summary()
    recommendation = recommend_forecasting_model(
        summary
    )

    anomaly_summary = {
        "known_closure_count": 247,
        "actionable_anomaly_count": 115,
        "positive_anomaly_count": 87,
        "negative_anomaly_count": 28,
        "actionable_anomaly_rate_percent": 6.5,
    }

    episode_summary = {
        "episode_count": 37,
        "episode_context_anomaly_hours": {
            "forecast_floor_positive_episode": 62,
            "rain_coincident_negative_episode": 22,
            "other_residual_episode": 31,
        },
        "top_ten_dates_share_percent": 80.87,
    }

    output_path = (
        tmp_path / "model_recommendation.md"
    )

    saved_path = (
        write_model_recommendation_report(
            recommendation=recommendation,
            model_summary=summary,
            anomaly_summary=anomaly_summary,
            episode_summary=episode_summary,
            output_path=output_path,
        )
    )

    contents = saved_path.read_text(
        encoding="utf-8"
    )

    assert saved_path.exists()
    assert "# Forecasting Model Recommendation" in contents
    assert "gradient_boosting_recursive" in contents
    assert "seasonal_naive_168" in contents
    assert "## Operational policy" in contents
    assert "Forecast-floor" in contents