"""Analyze selected-model errors using only inner cold-drug OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.cross_validation import OOF_COLUMNS, PRIMARY_COMPARISON_METRIC
from src.models.evaluation import (
    DEFAULT_DECISION_THRESHOLD,
    evaluate_binary_classification,
)


DEFAULT_TOP_N = 5
DEFAULT_MINIMUM_GROUP_SIZE = 20
DEFAULT_MINIMUM_RELEVANT_CLASS_COUNT = 5

ERROR_TYPES = (
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
)
ERROR_ROW_COLUMNS = (*OOF_COLUMNS, "y_pred", "error_type")
ENTITY_SUMMARY_COLUMNS = (
    "entity_type",
    "entity_id",
    "pair_count",
    "positive_count",
    "negative_count",
    "true_positive_count",
    "false_positive_count",
    "false_negative_count",
    "true_negative_count",
    "precision",
    "recall",
    "false_positive_rate",
    "false_negative_rate",
    "mean_positive_probability",
)


class ErrorAnalysisError(ValueError):
    """Raised when an OOF error analysis would be invalid or misleading."""


@dataclass(frozen=True)
class ErrorAnalysisResult:
    """Version-controlled summary plus ignored detailed local tables."""

    report: dict[str, Any]
    error_rows: pd.DataFrame
    entity_summary: pd.DataFrame


def _finite_float(value: object, name: str) -> float:
    """Return a finite numeric value with a compact validation message."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ErrorAnalysisError(f"{name} must be numeric.") from error

    if not np.isfinite(numeric_value):
        raise ErrorAnalysisError(f"{name} must be finite.")

    return numeric_value


def _positive_integer(value: object, name: str) -> int:
    """Return an integer greater than zero without truncating decimals."""
    numeric_value = _finite_float(value, name)

    if not numeric_value.is_integer() or numeric_value < 1:
        raise ErrorAnalysisError(f"{name} must be a positive integer.")

    return int(numeric_value)


def _threshold(value: object) -> float:
    """Validate the fixed probability threshold used for error categories."""
    threshold = _finite_float(value, "decision_threshold")

    if not 0.0 <= threshold <= 1.0:
        raise ErrorAnalysisError(
            "decision_threshold must lie between 0 and 1."
        )

    return threshold


def select_model_from_inner_cv_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Select exactly one candidate by mean inner-fold AP, never pooled AP."""
    if not isinstance(summary, dict):
        raise ErrorAnalysisError("Inner-CV summary must be a JSON object.")

    if summary.get("outer_policy") != "cold_drug":
        raise ErrorAnalysisError("Inner-CV policy must be cold_drug.")

    if summary.get("cv_scope") != "frozen_outer_training_partition_only":
        raise ErrorAnalysisError(
            "Inner-CV summary must be limited to outer-training data."
        )

    if summary.get("outer_test_partition_used") is not False:
        raise ErrorAnalysisError(
            "Inner-CV summary must confirm that the outer test partition was "
            "not used."
        )

    if summary.get("primary_comparison_metric") != (
        PRIMARY_COMPARISON_METRIC
    ):
        raise ErrorAnalysisError(
            "Model selection must use the prespecified average-precision "
            "metric."
        )

    label_column = summary.get("label_column")
    if not isinstance(label_column, str) or not label_column.strip():
        raise ErrorAnalysisError("Inner-CV summary is missing label_column.")

    n_splits = _positive_integer(summary.get("n_splits"), "n_splits")
    pair_count = _positive_integer(
        summary.get("input_pair_count"),
        "input_pair_count",
    )
    drug_count = _positive_integer(
        summary.get("input_drug_count"),
        "input_drug_count",
    )
    results = summary.get("model_results")

    if not isinstance(results, list) or len(results) < 2:
        raise ErrorAnalysisError("Inner-CV summary needs at least two models.")

    candidates: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()

    for result in results:
        if not isinstance(result, dict):
            raise ErrorAnalysisError("Each model result must be a JSON object.")

        model_id = result.get("model_id")
        model_name = result.get("model_name")
        fold_metric_summary = result.get("fold_metric_summary")
        metric_summary = (
            fold_metric_summary.get(PRIMARY_COMPARISON_METRIC)
            if isinstance(fold_metric_summary, dict)
            else None
        )

        if not isinstance(model_id, str) or not model_id.strip():
            raise ErrorAnalysisError("A candidate is missing model_id.")
        if model_id in seen_model_ids:
            raise ErrorAnalysisError("Candidate model_id values must be unique.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ErrorAnalysisError(f"{model_id} is missing model_name.")
        if not isinstance(metric_summary, dict):
            raise ErrorAnalysisError(
                f"{model_id} lacks a fold average-precision summary."
            )

        seen_model_ids.add(model_id)
        candidates.append(
            {
                "model_id": model_id,
                "model_name": model_name,
                "mean": _finite_float(
                    metric_summary.get("mean"),
                    f"{model_id} mean average precision",
                ),
                "standard_deviation": _finite_float(
                    metric_summary.get("standard_deviation"),
                    f"{model_id} average-precision standard deviation",
                ),
                "oof_prediction_count": _positive_integer(
                    result.get("oof_prediction_count"),
                    f"{model_id} oof_prediction_count",
                ),
            }
        )

    ranked = sorted(
        candidates,
        key=lambda candidate: (-candidate["mean"], candidate["model_id"]),
    )
    selected, runner_up = ranked[:2]

    if np.isclose(
        selected["mean"], runner_up["mean"], rtol=0.0, atol=1e-12
    ):
        raise ErrorAnalysisError(
            "The leading candidates are tied on mean average precision."
        )

    if selected["oof_prediction_count"] != pair_count:
        raise ErrorAnalysisError(
            "Selected-model OOF count does not match the inner-CV pair count."
        )

    return {
        "selection_scope": "inner_cold_drug_cv_only",
        "selection_statistic": "unweighted_mean_across_grouped_folds",
        "selection_metric": PRIMARY_COMPARISON_METRIC,
        "selected_model_id": selected["model_id"],
        "selected_model_name": selected["model_name"],
        "selected_mean": selected["mean"],
        "selected_standard_deviation": selected["standard_deviation"],
        "runner_up_model_id": runner_up["model_id"],
        "runner_up_model_name": runner_up["model_name"],
        "runner_up_mean": runner_up["mean"],
        "runner_up_standard_deviation": runner_up["standard_deviation"],
        "mean_difference_from_runner_up": (
            selected["mean"] - runner_up["mean"]
        ),
        "label_column": label_column,
        "n_splits": n_splits,
        "input_pair_count": pair_count,
        "input_drug_count": drug_count,
    }


def _integer_column(frame: pd.DataFrame, column_name: str) -> None:
    """Validate one numeric column and replace it with integer values."""
    try:
        values = pd.to_numeric(frame[column_name], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ErrorAnalysisError(f"{column_name} must be numeric.") from error

    if not np.isfinite(values.to_numpy()).all() or not np.equal(
        values.to_numpy(), np.floor(values.to_numpy())
    ).all():
        raise ErrorAnalysisError(f"{column_name} must contain finite integers.")

    frame[column_name] = values.astype("int64")


def select_oof_predictions(
    all_predictions: pd.DataFrame,
    selection: dict[str, Any],
) -> pd.DataFrame:
    """Validate and retain just the selected candidate's OOF predictions."""
    if tuple(all_predictions.columns) != OOF_COLUMNS:
        raise ErrorAnalysisError(
            "OOF prediction columns do not match the frozen CV contract."
        )

    selected = all_predictions.loc[
        all_predictions["model_id"].eq(selection["selected_model_id"])
    ].copy()

    if len(selected) != selection["input_pair_count"]:
        raise ErrorAnalysisError(
            "Selected OOF prediction count does not match the inner-CV "
            "summary."
        )

    if selected["model_name"].nunique() != 1 or (
        selected["model_name"].iloc[0] != selection["selected_model_name"]
    ):
        raise ErrorAnalysisError("Selected OOF model name is inconsistent.")

    for column_name in ("drug_id", "target_id"):
        if selected[column_name].isna().any():
            raise ErrorAnalysisError(f"{column_name} must not be missing.")
        selected[column_name] = selected[column_name].astype(str).str.strip()
        if selected[column_name].eq("").any():
            raise ErrorAnalysisError(f"{column_name} must not be empty.")

    for column_name in (
        "observed_pair_index",
        "fold_index",
        "fit_random_state",
        "y_true",
    ):
        _integer_column(selected, column_name)

    if selected["observed_pair_index"].duplicated().any():
        raise ErrorAnalysisError("Selected OOF pairs must be unique.")
    if not selected["y_true"].isin((0, 1)).all():
        raise ErrorAnalysisError("Selected OOF labels must be binary.")
    if selected["y_true"].nunique() != 2:
        raise ErrorAnalysisError("Selected OOF data must contain both classes.")
    if set(selected["fold_index"]) != set(range(selection["n_splits"])):
        raise ErrorAnalysisError("Selected OOF data do not cover every fold.")
    if selected["drug_id"].nunique() != selection["input_drug_count"]:
        raise ErrorAnalysisError("Selected OOF drug count is inconsistent.")

    try:
        scores = pd.to_numeric(
            selected["positive_probability"], errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as error:
        raise ErrorAnalysisError(
            "positive_probability must be numeric."
        ) from error

    if not np.isfinite(scores.to_numpy()).all() or (
        (scores < 0.0) | (scores > 1.0)
    ).any():
        raise ErrorAnalysisError(
            "positive_probability must be finite and lie between 0 and 1."
        )

    selected["positive_probability"] = scores
    return selected.sort_values(
        "observed_pair_index", kind="stable"
    ).reset_index(drop=True)


def build_error_rows(
    predictions: pd.DataFrame,
    *,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> pd.DataFrame:
    """Classify every OOF prediction at the pre-specified fixed threshold."""
    threshold = _threshold(decision_threshold)
    rows = predictions.copy()
    rows["y_pred"] = (
        rows["positive_probability"] >= threshold
    ).astype("int8")
    rows["error_type"] = np.select(
        [
            rows["y_true"].eq(0) & rows["y_pred"].eq(0),
            rows["y_true"].eq(0) & rows["y_pred"].eq(1),
            rows["y_true"].eq(1) & rows["y_pred"].eq(0),
            rows["y_true"].eq(1) & rows["y_pred"].eq(1),
        ],
        ERROR_TYPES,
        default="unclassified",
    )

    if rows["error_type"].eq("unclassified").any():
        raise ErrorAnalysisError("Could not classify every OOF prediction.")

    return rows.loc[:, list(ERROR_ROW_COLUMNS)]


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate, retaining undefined rates as JSON-safe null values."""
    return None if denominator == 0 else float(numerator / denominator)


def _counts(frame: pd.DataFrame) -> dict[str, int | float | None]:
    """Summarize confusion categories for one fold, drug, target, or bin."""
    counts = frame["error_type"].value_counts()
    true_negative = int(counts.get("true_negative", 0))
    false_positive = int(counts.get("false_positive", 0))
    false_negative = int(counts.get("false_negative", 0))
    true_positive = int(counts.get("true_positive", 0))
    pair_count = int(len(frame))
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive

    if (
        true_negative
        + false_positive
        + false_negative
        + true_positive
        != pair_count
    ):
        raise ErrorAnalysisError("Confusion categories do not cover a group.")

    return {
        "pair_count": pair_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "true_negative_count": true_negative,
        "precision": _rate(true_positive, true_positive + false_positive),
        "recall": _rate(true_positive, positive_count),
        "false_positive_rate": _rate(false_positive, negative_count),
        "false_negative_rate": _rate(false_negative, positive_count),
        "mean_positive_probability": float(
            frame["positive_probability"].mean()
        ) if pair_count else None,
    }


def _entity_summary(
    rows: pd.DataFrame,
    entity_type: str,
    entity_column: str,
) -> pd.DataFrame:
    """Generate descriptive error slices by drug or target identifier."""
    records = []

    for entity_id, group in rows.groupby(entity_column, sort=True):
        records.append(
            {
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                **_counts(group),
            }
        )

    return pd.DataFrame(records, columns=ENTITY_SUMMARY_COLUMNS)


def _fold_summaries(rows: pd.DataFrame) -> list[dict[str, object]]:
    """Keep fold-specific summaries separate from pooled descriptive values."""
    records = []

    for (fold_index, seed), group in rows.groupby(
        ["fold_index", "fit_random_state"], sort=True
    ):
        records.append(
            {
                "fold_index": int(fold_index),
                "fit_random_state": int(seed),
                **_counts(group),
            }
        )

    return records


def _probability_bins(rows: pd.DataFrame) -> list[dict[str, object]]:
    """Describe, rather than calibrate, labels across ten fixed score bins."""
    scores = rows["positive_probability"].to_numpy(dtype=float)
    records = []

    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        final_bin = index == 9
        mask = (scores >= lower) & (
            scores <= upper if final_bin else scores < upper
        )
        group = rows.loc[mask]
        record = _counts(group)
        record.update(
            {
                "probability_bin": (
                    f"[{lower:.1f}, {upper:.1f}]"
                    if final_bin
                    else f"[{lower:.1f}, {upper:.1f})"
                ),
                "lower_bound": lower,
                "upper_bound": upper,
                "observed_positive_rate": _rate(
                    int(record["positive_count"]),
                    int(record["pair_count"]),
                ),
            }
        )
        records.append(record)

    return records


def _top_entities(
    summary: pd.DataFrame,
    *,
    rate_column: str,
    relevant_count_column: str,
    top_n: int,
    minimum_group_size: int,
    minimum_relevant_class_count: int,
) -> list[dict[str, object]]:
    """Return qualified descriptive examples; do not interpret them biologically."""
    qualified = summary.loc[
        summary["pair_count"].ge(minimum_group_size)
        & summary[relevant_count_column].ge(minimum_relevant_class_count)
        & summary[rate_column].notna()
    ]
    columns = (
        "entity_type",
        "entity_id",
        "pair_count",
        "positive_count",
        "negative_count",
        "false_positive_count",
        "false_negative_count",
        rate_column,
        "mean_positive_probability",
    )

    return _records(
        qualified.sort_values(
            [rate_column, relevant_count_column, "entity_id"],
            ascending=[False, False, True],
            kind="stable",
        ).head(top_n).loc[:, list(columns)]
    )


def _json_value(value: object) -> object:
    """Convert pandas/NumPy scalar values to strict JSON values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return strict JSON-safe table records."""
    return [
        {name: _json_value(value) for name, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def run_error_analysis(
    inner_cv_summary: dict[str, Any],
    all_oof_predictions: pd.DataFrame,
    *,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
    top_n: int = DEFAULT_TOP_N,
    minimum_group_size: int = DEFAULT_MINIMUM_GROUP_SIZE,
    minimum_relevant_class_count: int = (
        DEFAULT_MINIMUM_RELEVANT_CLASS_COUNT
    ),
) -> ErrorAnalysisResult:
    """Analyze only OOF data after locking the inner-CV selection decision."""
    threshold = _threshold(decision_threshold)
    top_n = _positive_integer(top_n, "top_n")
    minimum_group_size = _positive_integer(
        minimum_group_size, "minimum_group_size"
    )
    minimum_relevant_class_count = _positive_integer(
        minimum_relevant_class_count, "minimum_relevant_class_count"
    )

    selection = select_model_from_inner_cv_summary(inner_cv_summary)
    selected = select_oof_predictions(all_oof_predictions, selection)
    error_rows = build_error_rows(selected, decision_threshold=threshold)

    pooled_metrics = evaluate_binary_classification(
        error_rows["y_true"],
        error_rows["positive_probability"],
        decision_threshold=threshold,
    )
    drug_summary = _entity_summary(error_rows, "drug", "drug_id")
    target_summary = _entity_summary(error_rows, "target", "target_id")
    entity_summary = pd.concat(
        [drug_summary, target_summary], ignore_index=True
    ).loc[:, ENTITY_SUMMARY_COLUMNS]

    report = {
        "analysis_scope": "inner_cold_drug_out_of_fold_predictions_only",
        "outer_test_partition_used": False,
        "label_column": selection["label_column"],
        "selected_model": selection,
        "decision_threshold": threshold,
        "input_pair_count": int(len(error_rows)),
        "input_drug_count": int(error_rows["drug_id"].nunique()),
        "input_target_count": int(error_rows["target_id"].nunique()),
        "error_type_counts": {
            name: int(error_rows["error_type"].eq(name).sum())
            for name in ERROR_TYPES
        },
        "pooled_oof_metrics_descriptive_only": pooled_metrics.to_dict(),
        "fold_summaries": _fold_summaries(error_rows),
        "probability_bin_summaries": _probability_bins(error_rows),
        "entity_analysis": {
            "drug_group_count": int(len(drug_summary)),
            "target_group_count": int(len(target_summary)),
            "minimum_pair_count_for_ranked_examples": minimum_group_size,
            "minimum_relevant_class_count_for_ranked_examples": (
                minimum_relevant_class_count
            ),
            "top_n": top_n,
            "highest_false_negative_rate_drugs": _top_entities(
                drug_summary,
                rate_column="false_negative_rate",
                relevant_count_column="positive_count",
                top_n=top_n,
                minimum_group_size=minimum_group_size,
                minimum_relevant_class_count=minimum_relevant_class_count,
            ),
            "highest_false_positive_rate_drugs": _top_entities(
                drug_summary,
                rate_column="false_positive_rate",
                relevant_count_column="negative_count",
                top_n=top_n,
                minimum_group_size=minimum_group_size,
                minimum_relevant_class_count=minimum_relevant_class_count,
            ),
            "highest_false_negative_rate_targets": _top_entities(
                target_summary,
                rate_column="false_negative_rate",
                relevant_count_column="positive_count",
                top_n=top_n,
                minimum_group_size=minimum_group_size,
                minimum_relevant_class_count=minimum_relevant_class_count,
            ),
            "highest_false_positive_rate_targets": _top_entities(
                target_summary,
                rate_column="false_positive_rate",
                relevant_count_column="negative_count",
                top_n=top_n,
                minimum_group_size=minimum_group_size,
                minimum_relevant_class_count=minimum_relevant_class_count,
            ),
        },
        "interpretation_limits": [
            "Rows compare predictions with Davis threshold labels; they do not "
            "verify laboratory binding outcomes.",
            "Pooled OOF scores come from separately fitted folds, so pooled "
            "metrics and score bins are descriptive rather than selection "
            "criteria.",
            "Entity-level error patterns do not establish biological mechanisms "
            "or causal effects.",
            "This analysis must not change the selected model, tune a threshold, "
            "or re-open the frozen outer holdout.",
        ],
    }

    return ErrorAnalysisResult(report, error_rows, entity_summary)


def _write_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a deterministic local CSV table."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g")
    return path


def write_error_rows(rows: pd.DataFrame, output_path: str | Path) -> Path:
    """Write ignored pair-level errors with a checked column contract."""
    if tuple(rows.columns) != ERROR_ROW_COLUMNS or rows.empty:
        raise ErrorAnalysisError("Invalid or empty error-row table.")

    return _write_csv(rows, output_path)


def write_entity_summary(
    summary: pd.DataFrame, output_path: str | Path
) -> Path:
    """Write ignored drug/target error summaries with a checked contract."""
    if tuple(summary.columns) != ENTITY_SUMMARY_COLUMNS or summary.empty:
        raise ErrorAnalysisError("Invalid or empty entity-summary table.")

    return _write_csv(summary, output_path)


def write_error_analysis_summary(
    result: ErrorAnalysisResult, output_path: str | Path
) -> Path:
    """Write a compact version-controlled JSON report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """Run error analysis from the frozen inner-CV evidence and local OOF CSV."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze selected inner cold-drug OOF errors without using the "
            "outer test partition."
        )
    )
    parser.add_argument(
        "--inner-cv-summary",
        type=Path,
        default=Path("reports/davis_inner_cold_drug_cv.json"),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path(
            "data/interim/davis_inner_cold_drug_oof_predictions.csv"
        ),
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--minimum-group-size",
        type=int,
        default=DEFAULT_MINIMUM_GROUP_SIZE,
    )
    parser.add_argument(
        "--minimum-relevant-class-count",
        type=int,
        default=DEFAULT_MINIMUM_RELEVANT_CLASS_COUNT,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "reports/davis_random_forest_inner_cv_error_analysis.json"
        ),
    )
    parser.add_argument(
        "--error-rows-output",
        type=Path,
        default=Path(
            "data/interim/davis_random_forest_inner_cv_error_rows.csv"
        ),
    )
    parser.add_argument(
        "--entity-summary-output",
        type=Path,
        default=Path(
            "data/interim/davis_random_forest_inner_cv_entity_summary.csv"
        ),
    )
    args = parser.parse_args(argv)

    try:
        summary = json.loads(
            args.inner_cv_summary.read_text(encoding="utf-8")
        )
        predictions = pd.read_csv(
            args.oof_predictions,
            dtype={
                "model_id": str,
                "model_name": str,
                "drug_id": str,
                "target_id": str,
            },
        )
        result = run_error_analysis(
            summary,
            predictions,
            decision_threshold=args.decision_threshold,
            top_n=args.top_n,
            minimum_group_size=args.minimum_group_size,
            minimum_relevant_class_count=(
                args.minimum_relevant_class_count
            ),
        )
        summary_path = write_error_analysis_summary(
            result, args.summary_output
        )
        error_rows_path = write_error_rows(
            result.error_rows, args.error_rows_output
        )
        entity_summary_path = write_entity_summary(
            result.entity_summary, args.entity_summary_output
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        print(f"Error analysis failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.report, indent=2, sort_keys=True))
    print(f"Error-analysis summary written to: {summary_path}")
    print(f"Detailed error rows written to: {error_rows_path}")
    print(f"Entity summary written to: {entity_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())