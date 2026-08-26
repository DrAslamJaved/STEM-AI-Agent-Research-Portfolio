"""Transparent evidence-based forecasting-model recommendation."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from time_series_agent.exceptions import (
    RecommendationError,
)


REQUIRED_SUMMARY_COLUMNS = {
    "model",
    "folds",
    "mean_mae",
    "std_mae",
    "mean_rmse",
    "mean_smape",
    "mean_mase",
    "mae_fold_wins",
    "mae_rank",
    "total_raw_negative_forecasts",
}

NUMERIC_SUMMARY_COLUMNS = {
    "folds",
    "mean_mae",
    "std_mae",
    "mean_rmse",
    "mean_smape",
    "mean_mase",
    "mae_fold_wins",
    "mae_rank",
    "total_raw_negative_forecasts",
}

MODEL_COMPLEXITY = {
    "mean": "low",
    "naive": "low",
    "seasonal_naive_24": "low",
    "seasonal_naive_168": "low",
    "holt_winters_24": "medium",
    "gradient_boosting_recursive": "high",
}


@dataclass(frozen=True)
class ModelRecommendation:
    """Structured forecasting-model decision."""

    selected_model: str
    fallback_model: str
    benchmark_model: str
    best_accuracy_candidate: str
    selection_status: str
    primary_metric: str
    minimum_required_mae_improvement_percentage: float
    observed_mae_improvement_percentage: float
    observed_rmse_improvement_percentage: float
    candidate_mean_mae: float
    benchmark_mean_mae: float
    candidate_mean_rmse: float
    benchmark_mean_rmse: float
    candidate_mae_standard_deviation: float
    benchmark_mae_standard_deviation: float
    candidate_fold_wins: int
    benchmark_fold_wins: int
    candidate_total_raw_negative_forecasts: int
    candidate_raw_negative_rate_percentage: float
    candidate_meets_mae_improvement_rule: bool
    candidate_meets_rmse_rule: bool
    candidate_evaluated_on_same_folds: bool
    selected_model_complexity: str
    fallback_model_complexity: str
    warnings: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible recommendation."""
        return {
            "selected_model": self.selected_model,
            "fallback_model": self.fallback_model,
            "benchmark_model": self.benchmark_model,
            "best_accuracy_candidate": (
                self.best_accuracy_candidate
            ),
            "selection_status": self.selection_status,
            "primary_metric": self.primary_metric,
            "minimum_required_mae_improvement_percentage": (
                self.minimum_required_mae_improvement_percentage
            ),
            "observed_mae_improvement_percentage": (
                self.observed_mae_improvement_percentage
            ),
            "observed_rmse_improvement_percentage": (
                self.observed_rmse_improvement_percentage
            ),
            "candidate_mean_mae": (
                self.candidate_mean_mae
            ),
            "benchmark_mean_mae": (
                self.benchmark_mean_mae
            ),
            "candidate_mean_rmse": (
                self.candidate_mean_rmse
            ),
            "benchmark_mean_rmse": (
                self.benchmark_mean_rmse
            ),
            "candidate_mae_standard_deviation": (
                self.candidate_mae_standard_deviation
            ),
            "benchmark_mae_standard_deviation": (
                self.benchmark_mae_standard_deviation
            ),
            "candidate_fold_wins": (
                self.candidate_fold_wins
            ),
            "benchmark_fold_wins": (
                self.benchmark_fold_wins
            ),
            "candidate_total_raw_negative_forecasts": (
                self.candidate_total_raw_negative_forecasts
            ),
            "candidate_raw_negative_rate_percentage": (
                self.candidate_raw_negative_rate_percentage
            ),
            "decision_checks": {
                "candidate_meets_mae_improvement_rule": (
                    self.candidate_meets_mae_improvement_rule
                ),
                "candidate_meets_rmse_rule": (
                    self.candidate_meets_rmse_rule
                ),
                "candidate_evaluated_on_same_folds": (
                    self.candidate_evaluated_on_same_folds
                ),
            },
            "selected_model_complexity": (
                self.selected_model_complexity
            ),
            "fallback_model_complexity": (
                self.fallback_model_complexity
            ),
            "warnings": list(self.warnings),
            "reason": self.reason,
        }


def _validate_policy_parameter(
    value: float,
    name: str,
) -> float:
    """Validate a nonnegative finite policy value."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
        or value < 0
    ):
        raise RecommendationError(
            f"'{name}' must be a nonnegative finite number."
        )

    return float(value)


def _validate_prediction_count(
    total_predictions_per_model: int,
) -> int:
    """Validate the number of evaluated predictions."""
    if (
        not isinstance(
            total_predictions_per_model,
            int,
        )
        or isinstance(
            total_predictions_per_model,
            bool,
        )
        or total_predictions_per_model <= 0
    ):
        raise RecommendationError(
            "'total_predictions_per_model' must be a "
            "positive integer."
        )

    return total_predictions_per_model


def _validate_model_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Validate comparable model-summary evidence."""
    if not isinstance(summary, pd.DataFrame):
        raise RecommendationError(
            "'summary' must be a pandas DataFrame."
        )

    if summary.empty:
        raise RecommendationError(
            "Model summary cannot be empty."
        )

    missing_columns = sorted(
        REQUIRED_SUMMARY_COLUMNS
        - set(summary.columns)
    )

    if missing_columns:
        raise RecommendationError(
            "Model summary is missing columns: "
            + ", ".join(missing_columns)
        )

    validated = summary.copy(deep=True)

    model_names = validated["model"]

    if model_names.isna().any():
        raise RecommendationError(
            "Model names cannot be missing."
        )

    model_names = model_names.astype(str).str.strip()

    if model_names.eq("").any():
        raise RecommendationError(
            "Model names cannot be empty."
        )

    if model_names.duplicated().any():
        raise RecommendationError(
            "Model names must be unique."
        )

    validated["model"] = model_names

    for column in NUMERIC_SUMMARY_COLUMNS:
        numeric_values = pd.to_numeric(
            validated[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            raise RecommendationError(
                f"Column '{column}' must be numeric "
                "and complete."
            )

        if not np.isfinite(
            numeric_values.to_numpy(
                dtype="float64"
            )
        ).all():
            raise RecommendationError(
                f"Column '{column}' must be finite."
            )

        validated[column] = numeric_values

    nonnegative_columns = [
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "mean_smape",
        "mean_mase",
        "mae_fold_wins",
        "total_raw_negative_forecasts",
    ]

    if validated[
        nonnegative_columns
    ].lt(0).any().any():
        raise RecommendationError(
            "Model metrics and counts cannot be negative."
        )

    if validated["folds"].le(0).any():
        raise RecommendationError(
            "Every model must have at least one fold."
        )

    if validated["folds"].nunique() != 1:
        raise RecommendationError(
            "All models must be evaluated on the same "
            "number of folds."
        )

    return validated


def recommend_forecasting_model(
    summary: pd.DataFrame,
    benchmark_model: str = "seasonal_naive_168",
    minimum_mae_improvement_percentage: float = 2.0,
    total_predictions_per_model: int = 2016,
) -> ModelRecommendation:
    """Recommend a model using transparent decision gates."""
    validated_summary = _validate_model_summary(
        summary
    )

    improvement_requirement = (
        _validate_policy_parameter(
            minimum_mae_improvement_percentage,
            "minimum_mae_improvement_percentage",
        )
    )

    prediction_count = _validate_prediction_count(
        total_predictions_per_model
    )

    if (
        not isinstance(benchmark_model, str)
        or not benchmark_model.strip()
    ):
        raise RecommendationError(
            "'benchmark_model' must be a nonempty string."
        )

    benchmark_name = benchmark_model.strip()

    benchmark_matches = validated_summary.loc[
        validated_summary["model"].eq(
            benchmark_name
        )
    ]

    if benchmark_matches.empty:
        raise RecommendationError(
            f"Benchmark model '{benchmark_name}' "
            "is not present."
        )

    ranked = validated_summary.sort_values(
        by=["mean_mae", "mean_rmse", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    candidate_row = ranked.iloc[0]
    benchmark_row = benchmark_matches.iloc[0]

    candidate_name = str(
        candidate_row["model"]
    )

    benchmark_mae = float(
        benchmark_row["mean_mae"]
    )
    candidate_mae = float(
        candidate_row["mean_mae"]
    )

    if benchmark_mae <= 0:
        raise RecommendationError(
            "Benchmark MAE must be positive."
        )

    benchmark_rmse = float(
        benchmark_row["mean_rmse"]
    )
    candidate_rmse = float(
        candidate_row["mean_rmse"]
    )

    mae_improvement = float(
        100
        * (benchmark_mae - candidate_mae)
        / benchmark_mae
    )

    if benchmark_rmse == 0:
        rmse_improvement = 0.0
    else:
        rmse_improvement = float(
            100
            * (benchmark_rmse - candidate_rmse)
            / benchmark_rmse
        )

    candidate_is_benchmark = (
        candidate_name == benchmark_name
    )

    meets_mae_rule = bool(
        candidate_is_benchmark
        or mae_improvement
        >= improvement_requirement
    )

    meets_rmse_rule = bool(
        candidate_rmse <= benchmark_rmse
    )

    if candidate_is_benchmark:
        selected_model = benchmark_name
        selection_status = "benchmark_already_best"
        reason = (
            "The weekly seasonal benchmark already has "
            "the lowest mean MAE."
        )
    elif meets_mae_rule and meets_rmse_rule:
        selected_model = candidate_name
        selection_status = "candidate_selected"
        reason = (
            f"{candidate_name} improves mean MAE by "
            f"{mae_improvement:.2f}% and does not worsen "
            "mean RMSE relative to the benchmark."
        )
    else:
        selected_model = benchmark_name
        selection_status = "benchmark_retained"
        reason = (
            f"{candidate_name} has the lowest mean MAE "
            "but does not satisfy every operational "
            "selection gate."
        )

    if selected_model != benchmark_name:
        fallback_model = benchmark_name
    else:
        alternative_rows = ranked.loc[
            ~ranked["model"].eq(benchmark_name)
        ]

        if alternative_rows.empty:
            fallback_model = benchmark_name
        else:
            fallback_model = str(
                alternative_rows.iloc[0]["model"]
            )

    candidate_std = float(
        candidate_row["std_mae"]
    )
    benchmark_std = float(
        benchmark_row["std_mae"]
    )

    candidate_wins = int(
        candidate_row["mae_fold_wins"]
    )
    benchmark_wins = int(
        benchmark_row["mae_fold_wins"]
    )

    raw_negative_count = int(
        candidate_row[
            "total_raw_negative_forecasts"
        ]
    )

    raw_negative_rate = float(
        100
        * raw_negative_count
        / prediction_count
    )

    warnings: list[str] = []

    if candidate_std > benchmark_std:
        warnings.append(
            "The best-MAE candidate has greater "
            "fold-to-fold MAE variability than the benchmark."
        )

    if candidate_wins < benchmark_wins:
        warnings.append(
            "The best-MAE candidate wins fewer folds "
            "than the benchmark."
        )

    if raw_negative_count > 0:
        warnings.append(
            "The best-MAE candidate produces raw negative "
            "count forecasts and requires nonnegative clipping."
        )

    if not meets_mae_rule:
        warnings.append(
            "The candidate's MAE improvement is below "
            "the required policy threshold."
        )

    if not meets_rmse_rule:
        warnings.append(
            "The candidate worsens mean RMSE relative "
            "to the benchmark."
        )

    selected_complexity = MODEL_COMPLEXITY.get(
        selected_model,
        "unknown",
    )
    fallback_complexity = MODEL_COMPLEXITY.get(
        fallback_model,
        "unknown",
    )

    return ModelRecommendation(
        selected_model=selected_model,
        fallback_model=fallback_model,
        benchmark_model=benchmark_name,
        best_accuracy_candidate=candidate_name,
        selection_status=selection_status,
        primary_metric="mean_mae",
        minimum_required_mae_improvement_percentage=(
            improvement_requirement
        ),
        observed_mae_improvement_percentage=(
            mae_improvement
        ),
        observed_rmse_improvement_percentage=(
            rmse_improvement
        ),
        candidate_mean_mae=candidate_mae,
        benchmark_mean_mae=benchmark_mae,
        candidate_mean_rmse=candidate_rmse,
        benchmark_mean_rmse=benchmark_rmse,
        candidate_mae_standard_deviation=(
            candidate_std
        ),
        benchmark_mae_standard_deviation=(
            benchmark_std
        ),
        candidate_fold_wins=candidate_wins,
        benchmark_fold_wins=benchmark_wins,
        candidate_total_raw_negative_forecasts=(
            raw_negative_count
        ),
        candidate_raw_negative_rate_percentage=(
            raw_negative_rate
        ),
        candidate_meets_mae_improvement_rule=(
            meets_mae_rule
        ),
        candidate_meets_rmse_rule=(
            meets_rmse_rule
        ),
        candidate_evaluated_on_same_folds=True,
        selected_model_complexity=(
            selected_complexity
        ),
        fallback_model_complexity=(
            fallback_complexity
        ),
        warnings=tuple(warnings),
        reason=reason,
    )


def save_model_recommendation(
    recommendation: ModelRecommendation,
    output_path: str | Path,
) -> Path:
    """Save the structured recommendation to JSON."""
    if not isinstance(
        recommendation,
        ModelRecommendation,
    ):
        raise RecommendationError(
            "'recommendation' must be a "
            "ModelRecommendation."
        )

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            recommendation.to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return destination

def write_model_recommendation_report(
    recommendation: ModelRecommendation,
    model_summary: pd.DataFrame,
    anomaly_summary: dict[str, object],
    episode_summary: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Write a human-readable model recommendation."""
    if not isinstance(
        recommendation,
        ModelRecommendation,
    ):
        raise RecommendationError(
            "'recommendation' must be a "
            "ModelRecommendation."
        )

    validated_summary = _validate_model_summary(
        model_summary
    )

    if not isinstance(anomaly_summary, dict):
        raise RecommendationError(
            "'anomaly_summary' must be a dictionary."
        )

    if not isinstance(episode_summary, dict):
        raise RecommendationError(
            "'episode_summary' must be a dictionary."
        )

    required_anomaly_keys = {
        "known_closure_count",
        "actionable_anomaly_count",
        "positive_anomaly_count",
        "negative_anomaly_count",
        "actionable_anomaly_rate_percent",
    }

    missing_anomaly_keys = sorted(
        required_anomaly_keys
        - set(anomaly_summary)
    )

    if missing_anomaly_keys:
        raise RecommendationError(
            "Anomaly summary is missing keys: "
            + ", ".join(missing_anomaly_keys)
        )

    required_episode_keys = {
        "episode_count",
        "episode_context_anomaly_hours",
        "top_ten_dates_share_percent",
    }

    missing_episode_keys = sorted(
        required_episode_keys
        - set(episode_summary)
    )

    if missing_episode_keys:
        raise RecommendationError(
            "Episode summary is missing keys: "
            + ", ".join(missing_episode_keys)
        )

    decision = recommendation.to_dict()

    ranked = validated_summary.sort_values(
        by=["mean_mae", "mean_rmse", "model"],
        ascending=[True, True, True],
    )

    episode_hours = episode_summary[
        "episode_context_anomaly_hours"
    ]

    lines = [
        "# Forecasting Model Recommendation",
        "",
        "## Decision",
        "",
        (
            f"**Preferred model:** "
            f"`{recommendation.selected_model}`"
        ),
        "",
        (
            f"**Fallback model:** "
            f"`{recommendation.fallback_model}`"
        ),
        "",
        (
            f"**Decision status:** "
            f"`{recommendation.selection_status}`"
        ),
        "",
        recommendation.reason,
        "",
        "## Selection policy",
        "",
        (
            "The agent ranks models by mean MAE across identical "
            "expanding-window folds. A more complex candidate replaces "
            "the weekly seasonal-naive benchmark only when:"
        ),
        "",
        "1. its mean-MAE improvement is at least 2%;",
        "2. its mean RMSE is no worse than the benchmark;",
        "3. all models were evaluated over the same folds.",
        "",
        "| Decision check | Requirement | Observed | Result |",
        "|---|---:|---:|---|",
        (
            "| MAE improvement | "
            f">= {recommendation.minimum_required_mae_improvement_percentage:.2f}% | "
            f"{recommendation.observed_mae_improvement_percentage:.2f}% | "
            f"{'Pass' if recommendation.candidate_meets_mae_improvement_rule else 'Fail'} |"
        ),
        (
            "| RMSE improvement | No degradation | "
            f"{recommendation.observed_rmse_improvement_percentage:.2f}% | "
            f"{'Pass' if recommendation.candidate_meets_rmse_rule else 'Fail'} |"
        ),
        (
            "| Comparable folds | Same fold count | "
            f"{'Yes' if recommendation.candidate_evaluated_on_same_folds else 'No'} | "
            f"{'Pass' if recommendation.candidate_evaluated_on_same_folds else 'Fail'} |"
        ),
        "",
        "## Six-model evidence",
        "",
        (
            "| Rank | Model | Mean MAE | MAE standard deviation | "
            "Mean RMSE | Mean sMAPE | Mean MASE | Fold wins | "
            "Raw negative forecasts |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            "| "
            f"{int(row['mae_rank'])} | "
            f"{row['model']} | "
            f"{float(row['mean_mae']):.2f} | "
            f"{float(row['std_mae']):.2f} | "
            f"{float(row['mean_rmse']):.2f} | "
            f"{float(row['mean_smape']):.2f} | "
            f"{float(row['mean_mase']):.3f} | "
            f"{int(row['mae_fold_wins'])} | "
            f"{int(row['total_raw_negative_forecasts'])} |"
        )

    lines.extend(
        [
            "",
            "## Candidate strengths",
            "",
            (
                f"- Mean MAE improved by "
                f"{recommendation.observed_mae_improvement_percentage:.2f}% "
                "relative to the weekly benchmark."
            ),
            (
                f"- Mean RMSE improved by "
                f"{recommendation.observed_rmse_improvement_percentage:.2f}%."
            ),
            (
                f"- The selected model ranked first by the primary "
                f"metric, mean MAE."
            ),
            "",
            "## Candidate cautions",
            "",
        ]
    )

    if recommendation.warnings:
        for warning in recommendation.warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No policy cautions were recorded.")

    lines.extend(
        [
            (
                f"- Raw-negative forecast rate: "
                f"{recommendation.candidate_raw_negative_rate_percentage:.2f}%."
            ),
            (
                f"- Selected-model complexity: "
                f"{recommendation.selected_model_complexity}."
            ),
            (
                f"- Fallback-model complexity: "
                f"{recommendation.fallback_model_complexity}."
            ),
            "",
            "## Anomaly-monitoring evidence",
            "",
            (
                f"The residual agent identified "
                f"{anomaly_summary['actionable_anomaly_count']} actionable "
                "candidate hours, corresponding to "
                f"{float(anomaly_summary['actionable_anomaly_rate_percent']):.2f}% "
                "of nonclosure residuals."
            ),
            "",
            (
                f"These alerts form "
                f"{episode_summary['episode_count']} consecutive episodes."
            ),
            "",
            (
                "- Forecast-floor positive anomaly hours: "
                f"{episode_hours.get('forecast_floor_positive_episode', 0)}."
            ),
            (
                "- Rain-coincident negative anomaly hours: "
                f"{episode_hours.get('rain_coincident_negative_episode', 0)}."
            ),
            (
                "- Other residual anomaly hours: "
                f"{episode_hours.get('other_residual_episode', 0)}."
            ),
            (
                "- Known closures remain separate from actionable alerts: "
                f"{anomaly_summary['known_closure_count']} closure hours."
            ),
            "",
            (
                "Forecast-floor episodes demonstrate that the selected "
                "model can become trapped near zero during recursive "
                "multi-step prediction. This limitation supports retaining "
                "the weekly seasonal-naive fallback."
            ),
            "",
            "## Operational policy",
            "",
            (
                "1. Use recursive Gradient Boosting as the preferred "
                "forecasting model."
            ),
            (
                "2. Apply and report the nonnegative forecast constraint."
            ),
            (
                "3. Monitor raw-negative forecasts and forecast-floor "
                "episodes."
            ),
            (
                "4. Use weekly seasonal naive when Gradient Boosting "
                "cannot fit, predict, or produce acceptable diagnostics."
            ),
            (
                "5. Keep documented closures separate from unexpected "
                "anomaly alerts."
            ),
            (
                "6. Treat residual anomalies as candidates requiring "
                "contextual or domain review."
            ),
            "",
            "## Limitations",
            "",
            (
                "- The recommendation is based on one public dataset and "
                "12 weekly expanding-window folds."
            ),
            (
                "- The 2% selection threshold is an explicit operational "
                "policy rather than a universal statistical law."
            ),
            (
                "- No formal significance test has yet been applied to "
                "paired fold errors."
            ),
            (
                "- The selected model excludes unknown future weather."
            ),
            (
                "- The anomaly dataset contains no verified ground-truth "
                "anomaly labels."
            ),
            "",
            "## Machine-readable decision",
            "",
            "The complete structured decision is stored in "
            "`reports/metrics/model_recommendation.json`.",
            "",
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return destination