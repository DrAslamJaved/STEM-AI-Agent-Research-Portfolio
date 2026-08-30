import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.cross_validation import OOF_COLUMNS
from src.models.error_analysis import (
    ENTITY_SUMMARY_COLUMNS,
    ERROR_ROW_COLUMNS,
    ErrorAnalysisError,
    run_error_analysis,
    select_model_from_inner_cv_summary,
    write_entity_summary,
    write_error_analysis_summary,
    write_error_rows,
)


def synthetic_inner_cv_summary() -> dict[str, object]:
    return {
        "outer_policy": "cold_drug",
        "cv_scope": "frozen_outer_training_partition_only",
        "outer_test_partition_used": False,
        "label_column": "interaction_kd_le_1000_nM",
        "n_splits": 2,
        "input_pair_count": 8,
        "input_drug_count": 2,
        "primary_comparison_metric": "average_precision",
        "model_results": [
            {
                "model_id": "dummy_prior",
                "model_name": "DummyClassifier",
                "oof_prediction_count": 8,
                "fold_metric_summary": {
                    "average_precision": {
                        "mean": 0.20,
                        "standard_deviation": 0.01,
                    }
                },
                "pooled_oof_metrics": {"average_precision": 0.20},
            },
            {
                "model_id": "random_forest_balanced",
                "model_name": "RandomForestClassifier",
                "oof_prediction_count": 8,
                "fold_metric_summary": {
                    "average_precision": {
                        "mean": 0.50,
                        "standard_deviation": 0.10,
                    }
                },
                "pooled_oof_metrics": {"average_precision": 0.40},
            },
            {
                "model_id": "hist_gradient_boosting_balanced",
                "model_name": "HistGradientBoostingClassifier",
                "oof_prediction_count": 8,
                "fold_metric_summary": {
                    "average_precision": {
                        "mean": 0.49,
                        "standard_deviation": 0.20,
                    }
                },
                "pooled_oof_metrics": {"average_precision": 0.99},
            },
        ],
    }


def synthetic_oof_predictions() -> pd.DataFrame:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.90, 0.80, 0.10, 0.20, 0.10, 0.90, 0.70, 0.60]
    records: list[dict[str, object]] = []

    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        records.append(
            {
                "model_id": "random_forest_balanced",
                "model_name": "RandomForestClassifier",
                "fold_index": 0 if index < 4 else 1,
                "fit_random_state": 7 if index < 4 else 8,
                "observed_pair_index": index,
                "drug_id": "drug_A" if index < 4 else "drug_B",
                "target_id": f"target_{index % 4}",
                "y_true": label,
                "positive_probability": score,
            }
        )

    for index, label in enumerate(labels):
        records.append(
            {
                "model_id": "hist_gradient_boosting_balanced",
                "model_name": "HistGradientBoostingClassifier",
                "fold_index": 0 if index < 4 else 1,
                "fit_random_state": 7 if index < 4 else 8,
                "observed_pair_index": 100 + index,
                "drug_id": "drug_A" if index < 4 else "drug_B",
                "target_id": f"target_{index % 4}",
                "y_true": label,
                "positive_probability": 0.50,
            }
        )

    return pd.DataFrame(records, columns=OOF_COLUMNS)


@pytest.fixture(scope="module")
def error_analysis_result():
    return run_error_analysis(
        synthetic_inner_cv_summary(),
        synthetic_oof_predictions(),
        decision_threshold=0.50,
        top_n=2,
        minimum_group_size=2,
        minimum_relevant_class_count=1,
    )


def test_selection_uses_mean_fold_ap_not_pooled_oof_ap() -> None:
    selection = select_model_from_inner_cv_summary(
        synthetic_inner_cv_summary()
    )

    assert selection["selected_model_id"] == "random_forest_balanced"
    assert selection["runner_up_model_id"] == "hist_gradient_boosting_balanced"
    assert selection["selected_mean"] == 0.50
    assert selection["runner_up_mean"] == 0.49
    assert selection["mean_difference_from_runner_up"] == pytest.approx(0.01)


def test_error_categories_and_group_summaries(
    error_analysis_result,
) -> None:
    result = error_analysis_result
    report = result.report

    assert tuple(result.error_rows.columns) == ERROR_ROW_COLUMNS
    assert len(result.error_rows) == 8
    assert report["outer_test_partition_used"] is False
    assert report["error_type_counts"] == {
        "true_negative": 2,
        "false_positive": 2,
        "false_negative": 1,
        "true_positive": 3,
    }
    assert report["pooled_oof_metrics_descriptive_only"]["true_positive"] == 3
    assert len(report["fold_summaries"]) == 2
    assert len(report["probability_bin_summaries"]) == 10
    assert sum(
        row["pair_count"] for row in report["probability_bin_summaries"]
    ) == 8

    assert tuple(result.entity_summary.columns) == ENTITY_SUMMARY_COLUMNS
    assert len(result.entity_summary) == 6
    assert set(result.entity_summary["entity_type"]) == {"drug", "target"}


def test_outer_test_usage_is_rejected() -> None:
    summary = synthetic_inner_cv_summary()
    summary["outer_test_partition_used"] = True

    with pytest.raises(ErrorAnalysisError, match="outer test partition"):
        select_model_from_inner_cv_summary(summary)


def test_duplicate_selected_pairs_are_rejected() -> None:
    predictions = synthetic_oof_predictions()
    predictions = pd.concat(
        [predictions, predictions.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ErrorAnalysisError, match="prediction count"):
        run_error_analysis(synthetic_inner_cv_summary(), predictions)


def test_writers_preserve_summary_and_table_contracts(
    tmp_path: Path,
    error_analysis_result,
) -> None:
    result = error_analysis_result
    summary_path = write_error_analysis_summary(
        result,
        tmp_path / "error_analysis.json",
    )
    error_rows_path = write_error_rows(
        result.error_rows,
        tmp_path / "error_rows.csv",
    )
    entity_summary_path = write_entity_summary(
        result.entity_summary,
        tmp_path / "entity_summary.csv",
    )

    report = json.loads(summary_path.read_text(encoding="utf-8"))
    restored_rows = pd.read_csv(error_rows_path)
    restored_entities = pd.read_csv(entity_summary_path)

    assert report["selected_model"]["selected_model_id"] == (
        "random_forest_balanced"
    )
    assert tuple(restored_rows.columns) == ERROR_ROW_COLUMNS
    assert tuple(restored_entities.columns) == ENTITY_SUMMARY_COLUMNS