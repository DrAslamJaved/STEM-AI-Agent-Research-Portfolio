"""Run frozen-fold Davis binary-label threshold sensitivity analysis.

The 100 nM analysis is deliberately descriptive: it reuses the frozen inner
cold-drug folds, fixed candidate models, and fixed hyperparameters from the
primary 1,000 nM experiment. It never reopens model selection or evaluates the
outer cold-drug holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.features.representations import FEATURE_COLUMNS
from src.models.baselines import DUMMY_STRATEGY
from src.models.evaluation import (
    BinaryClassificationMetrics,
    DEFAULT_DECISION_THRESHOLD,
    evaluate_binary_classification,
)
from src.models.hist_gradient_boosting import (
    DEFAULT_CLASS_WEIGHT as HGB_CLASS_WEIGHT,
    DEFAULT_EARLY_STOPPING as HGB_EARLY_STOPPING,
    DEFAULT_L2_REGULARIZATION as HGB_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE as HGB_LEARNING_RATE,
    DEFAULT_MAX_DEPTH as HGB_MAX_DEPTH,
    DEFAULT_MAX_FEATURES as HGB_MAX_FEATURES,
    DEFAULT_MAX_ITER as HGB_MAX_ITER,
    DEFAULT_MAX_LEAF_NODES as HGB_MAX_LEAF_NODES,
    DEFAULT_MIN_SAMPLES_LEAF as HGB_MIN_SAMPLES_LEAF,
    build_hist_gradient_boosting_pipeline,
)
from src.models.logistic_regression import (
    DEFAULT_C as LOGISTIC_C,
    DEFAULT_L1_RATIO as LOGISTIC_L1_RATIO,
    DEFAULT_MAX_ITER as LOGISTIC_MAX_ITER,
    DEFAULT_SOLVER as LOGISTIC_SOLVER,
    build_logistic_pipeline,
)
from src.models.random_forest import (
    DEFAULT_CLASS_WEIGHT as RF_CLASS_WEIGHT,
    DEFAULT_MAX_DEPTH as RF_MAX_DEPTH,
    DEFAULT_MAX_FEATURES as RF_MAX_FEATURES,
    DEFAULT_MIN_SAMPLES_LEAF as RF_MIN_SAMPLES_LEAF,
    DEFAULT_N_ESTIMATORS as RF_N_ESTIMATORS,
    DEFAULT_N_JOBS as RF_N_JOBS,
    build_random_forest_pipeline,
)


DEFAULT_RANDOM_STATE = 20260830
DEFAULT_N_SPLITS = 5
PRIMARY_LABEL_COLUMN = "interaction_kd_le_1000_nM"
SENSITIVITY_LABEL_COLUMN = "interaction_kd_le_100_nM"
REFERENCE_MODEL_ID = "dummy_prior"
PRIMARY_SELECTED_MODEL_ID = "random_forest_balanced"
PRIMARY_SELECTION_METRIC = "average_precision"

SUMMARY_METRICS = (
    "average_precision",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
)

SENSITIVITY_OOF_COLUMNS = (
    "threshold_variant",
    "label_column",
    "model_id",
    "model_name",
    "fold_index",
    "fit_random_state",
    "observed_pair_index",
    "drug_id",
    "target_id",
    "y_true",
    "positive_probability",
)


class ThresholdSensitivityError(ValueError):
    """Raised when frozen-fold threshold sensitivity is not trustworthy."""


@dataclass(frozen=True)
class LabelVariant:
    """One pre-specified binary definition of a Davis positive interaction."""

    variant_id: str
    label_column: str
    kd_threshold_nM: float
    pKd_threshold: float


LABEL_VARIANTS = (
    LabelVariant(
        variant_id="primary_kd_le_1000_nM",
        label_column=PRIMARY_LABEL_COLUMN,
        kd_threshold_nM=1000.0,
        pKd_threshold=6.0,
    ),
    LabelVariant(
        variant_id="sensitivity_kd_le_100_nM",
        label_column=SENSITIVITY_LABEL_COLUMN,
        kd_threshold_nM=100.0,
        pKd_threshold=7.0,
    ),
)


@dataclass(frozen=True)
class FixedCandidate:
    """One fixed model configuration reused unchanged for both labels."""

    model_id: str
    model_name: str
    model_parameters: dict[str, Any]
    builder: Callable[[int], Any]


@dataclass(frozen=True)
class FrozenFoldDataset:
    """Outer-training rows and primary-CV fold assignments only."""

    features: pd.DataFrame
    metadata: pd.DataFrame
    fold_indices: np.ndarray
    labels: dict[str, pd.Series]
    feature_columns: tuple[str, ...]
    n_splits: int


@dataclass(frozen=True)
class ThresholdSensitivityRun:
    """Committed report and ignored OOF predictions from fixed-fold analysis."""

    report: dict[str, Any]
    oof_predictions: pd.DataFrame


def _positive_integer(value: object, name: str) -> int:
    """Validate a positive integer without silently truncating a decimal."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ThresholdSensitivityError(f"{name} must be an integer.") from error

    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ThresholdSensitivityError(f"{name} must be an integer.")

    integer_value = int(numeric_value)
    if integer_value < 1:
        raise ThresholdSensitivityError(f"{name} must be at least one.")

    return integer_value


def _normalise_observed_pair_index(
    table: pd.DataFrame,
    table_label: str,
) -> pd.DataFrame:
    """Return a copy with finite integer pair IDs and no duplicate rows."""
    if "observed_pair_index" not in table.columns:
        raise ThresholdSensitivityError(
            f"{table_label} is missing observed_pair_index."
        )

    normalized = table.copy()
    numeric_indices = pd.to_numeric(
        normalized["observed_pair_index"],
        errors="coerce",
    )

    if numeric_indices.isna().any():
        raise ThresholdSensitivityError(
            f"{table_label} has non-numeric observed_pair_index values."
        )

    values = numeric_indices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(
        values,
        np.floor(values),
    ).all():
        raise ThresholdSensitivityError(
            f"{table_label} observed_pair_index values must be finite integers."
        )

    normalized["observed_pair_index"] = numeric_indices.astype(np.int64)
    if normalized["observed_pair_index"].duplicated().any():
        raise ThresholdSensitivityError(
            f"{table_label} contains duplicate observed_pair_index values."
        )

    return normalized


def _normalise_identifiers(
    table: pd.DataFrame,
    table_label: str,
) -> pd.DataFrame:
    """Return a copy with valid string drug and target identifiers."""
    required_columns = {"drug_id", "target_id"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ThresholdSensitivityError(
            f"{table_label} is missing columns: {sorted(missing_columns)}"
        )

    normalized = table.copy()
    for column in ("drug_id", "target_id"):
        if normalized[column].isna().any():
            raise ThresholdSensitivityError(
                f"{table_label} has missing {column} values."
            )

        normalized[column] = normalized[column].astype(str).str.strip()
        if normalized[column].eq("").any():
            raise ThresholdSensitivityError(
                f"{table_label} has empty {column} values."
            )

    return normalized


def _validated_binary_labels(
    table: pd.DataFrame,
    label_column: str,
) -> pd.Series:
    """Return a strict integer binary label series from one table column."""
    if label_column not in table.columns:
        raise ThresholdSensitivityError(
            f"Feature table is missing label column: {label_column}"
        )

    labels = pd.to_numeric(table[label_column], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ThresholdSensitivityError(
            f"{label_column} must contain only binary 0/1 values."
        )

    return labels.astype("int8").reset_index(drop=True)


def load_inner_cv_summary(path: str | Path) -> dict[str, Any]:
    """Load the committed primary inner-CV contract before sensitivity runs."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ThresholdSensitivityError(
            f"Could not load frozen inner-CV summary: {source}"
        ) from error

    if not isinstance(payload, dict):
        raise ThresholdSensitivityError("Inner-CV summary must be a JSON object.")

    return payload


def _validate_inner_cv_contract(
    summary: dict[str, Any],
    *,
    n_splits: int,
    random_state: int,
) -> None:
    """Ensure the reference OOF file came from the frozen primary CV design."""
    expected_values = {
        "outer_policy": "cold_drug",
        "cv_scope": "frozen_outer_training_partition_only",
        "outer_test_partition_used": False,
        "label_column": PRIMARY_LABEL_COLUMN,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
    }

    for field, expected_value in expected_values.items():
        if summary.get(field) != expected_value:
            raise ThresholdSensitivityError(
                f"Frozen inner-CV contract has unexpected {field}: "
                f"{summary.get(field)!r}."
            )


def extract_frozen_fold_assignments(
    all_oof_predictions: pd.DataFrame,
    *,
    reference_model_id: str = REFERENCE_MODEL_ID,
) -> pd.DataFrame:
    """Extract structural folds without reading prior labels or probabilities."""
    required_columns = {
        "model_id",
        "fold_index",
        "observed_pair_index",
        "drug_id",
        "target_id",
    }
    missing_columns = required_columns.difference(all_oof_predictions.columns)
    if missing_columns:
        raise ThresholdSensitivityError(
            "Reference OOF table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    assignments = all_oof_predictions.loc[
        all_oof_predictions["model_id"].astype(str) == reference_model_id,
        [
            "fold_index",
            "observed_pair_index",
            "drug_id",
            "target_id",
        ],
    ].copy()

    if assignments.empty:
        raise ThresholdSensitivityError(
            f"Reference OOF table has no rows for {reference_model_id}."
        )

    assignments = _normalise_observed_pair_index(
        assignments,
        "Reference OOF assignments",
    )
    assignments = _normalise_identifiers(
        assignments,
        "Reference OOF assignments",
    )

    fold_values = pd.to_numeric(assignments["fold_index"], errors="coerce")
    if fold_values.isna().any():
        raise ThresholdSensitivityError(
            "Reference OOF assignments have non-numeric fold indices."
        )

    numeric_fold_values = fold_values.to_numpy(dtype=float)
    if not np.isfinite(numeric_fold_values).all() or not np.equal(
        numeric_fold_values,
        np.floor(numeric_fold_values),
    ).all():
        raise ThresholdSensitivityError(
            "Reference OOF fold indices must be finite integers."
        )

    assignments["fold_index"] = fold_values.astype(int)
    if (assignments["fold_index"] < 0).any():
        raise ThresholdSensitivityError(
            "Reference OOF fold indices must be non-negative."
        )

    return assignments.sort_values(
        "observed_pair_index",
        kind="stable",
    ).reset_index(drop=True)


def build_frozen_fold_dataset(
    feature_table: pd.DataFrame,
    reference_assignments: pd.DataFrame,
    frozen_inner_cv_summary: dict[str, Any],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FrozenFoldDataset:
    """Build data only for rows represented in the frozen primary OOF file."""
    checked_n_splits = _positive_integer(n_splits, "n_splits")
    if checked_n_splits < 2:
        raise ThresholdSensitivityError("n_splits must be at least two.")
    _validate_inner_cv_contract(
        frozen_inner_cv_summary,
        n_splits=checked_n_splits,
        random_state=random_state,
    )

    required_columns = {
        "observed_pair_index",
        "drug_id",
        "target_id",
        *FEATURE_COLUMNS,
        PRIMARY_LABEL_COLUMN,
        SENSITIVITY_LABEL_COLUMN,
    }
    missing_columns = required_columns.difference(feature_table.columns)
    if missing_columns:
        raise ThresholdSensitivityError(
            "Feature table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    features = _normalise_observed_pair_index(feature_table, "Feature table")
    features = _normalise_identifiers(features, "Feature table")
    assignments = extract_frozen_fold_assignments(reference_assignments)

    expected_folds = set(range(checked_n_splits))
    observed_folds = set(assignments["fold_index"])
    if observed_folds != expected_folds:
        raise ThresholdSensitivityError(
            "Reference OOF assignments do not contain exactly the expected "
            f"folds: {sorted(expected_folds)}."
        )

    fold_counts_by_drug = assignments.groupby("drug_id")["fold_index"].nunique()
    if not fold_counts_by_drug.eq(1).all():
        raise ThresholdSensitivityError(
            "A drug appears in more than one frozen validation fold."
        )

    indexed_features = features.set_index("observed_pair_index")
    missing_pair_ids = set(assignments["observed_pair_index"]).difference(
        indexed_features.index
    )
    if missing_pair_ids:
        raise ThresholdSensitivityError(
            "Feature table is missing pairs in frozen OOF assignments."
        )

    selected = indexed_features.loc[
        assignments["observed_pair_index"],
        [
            "drug_id",
            "target_id",
            PRIMARY_LABEL_COLUMN,
            SENSITIVITY_LABEL_COLUMN,
            *FEATURE_COLUMNS,
        ],
    ].reset_index()

    if not np.array_equal(
        selected["drug_id"].to_numpy(),
        assignments["drug_id"].to_numpy(),
    ) or not np.array_equal(
        selected["target_id"].to_numpy(),
        assignments["target_id"].to_numpy(),
    ):
        raise ThresholdSensitivityError(
            "Feature-table entity IDs do not match frozen OOF assignments."
        )

    matrix = selected.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ThresholdSensitivityError(
            "Frozen outer-training features must be finite numeric values."
        )

    labels = {
        variant.label_column: _validated_binary_labels(
            selected,
            variant.label_column,
        )
        for variant in LABEL_VARIANTS
    }

    for variant in LABEL_VARIANTS:
        if labels[variant.label_column].nunique() != 2:
            raise ThresholdSensitivityError(
                f"{variant.label_column} needs both classes in outer training."
            )

    expected_pair_count = frozen_inner_cv_summary.get("input_pair_count")
    if expected_pair_count != len(selected):
        raise ThresholdSensitivityError(
            "Frozen inner-CV input_pair_count does not match reference OOF rows."
        )

    expected_drug_count = frozen_inner_cv_summary.get("input_drug_count")
    if expected_drug_count != int(assignments["drug_id"].nunique()):
        raise ThresholdSensitivityError(
            "Frozen inner-CV input_drug_count does not match reference OOF rows."
        )

    return FrozenFoldDataset(
        features=selected.loc[:, FEATURE_COLUMNS].reset_index(drop=True).copy(),
        metadata=pd.DataFrame(
            {
                "observed_pair_index": selected["observed_pair_index"].to_numpy(),
                "drug_id": assignments["drug_id"].to_numpy(),
                "target_id": assignments["target_id"].to_numpy(),
            }
        ),
        fold_indices=assignments["fold_index"].to_numpy(dtype=int),
        labels=labels,
        feature_columns=tuple(FEATURE_COLUMNS),
        n_splits=checked_n_splits,
    )


def _build_dummy_prior(random_state: int) -> DummyClassifier:
    """Build the fixed empirical-prior baseline."""
    return DummyClassifier(
        strategy=DUMMY_STRATEGY,
        random_state=int(random_state),
    )


def _candidate_models() -> tuple[FixedCandidate, ...]:
    """Return the fixed candidate list from the original inner-CV protocol."""
    return (
        FixedCandidate(
            model_id="dummy_prior",
            model_name="DummyClassifier",
            model_parameters={
                "random_state_rule": "base_random_state + frozen_fold_index",
                "strategy": DUMMY_STRATEGY,
            },
            builder=_build_dummy_prior,
        ),
        FixedCandidate(
            model_id="logistic_regression_balanced",
            model_name="LogisticRegression",
            model_parameters={
                "C": LOGISTIC_C,
                "class_weight": "balanced",
                "l1_ratio": LOGISTIC_L1_RATIO,
                "max_iter": LOGISTIC_MAX_ITER,
                "random_state_rule": "base_random_state + frozen_fold_index",
                "regularization": "L2",
                "solver": LOGISTIC_SOLVER,
                "standardization": "StandardScaler",
            },
            builder=lambda state: build_logistic_pipeline(
                "balanced",
                random_state=state,
            ),
        ),
        FixedCandidate(
            model_id="random_forest_balanced",
            model_name="RandomForestClassifier",
            model_parameters={
                "class_weight": RF_CLASS_WEIGHT,
                "max_depth": RF_MAX_DEPTH,
                "max_features": RF_MAX_FEATURES,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "n_estimators": RF_N_ESTIMATORS,
                "n_jobs": RF_N_JOBS,
                "random_state_rule": "base_random_state + frozen_fold_index",
                "zero_variance_selector": "VarianceThreshold(threshold=0.0)",
            },
            builder=lambda state: build_random_forest_pipeline(
                random_state=state
            ),
        ),
        FixedCandidate(
            model_id="hist_gradient_boosting_balanced",
            model_name="HistGradientBoostingClassifier",
            model_parameters={
                "class_weight": HGB_CLASS_WEIGHT,
                "early_stopping": HGB_EARLY_STOPPING,
                "l2_regularization": HGB_L2_REGULARIZATION,
                "learning_rate": HGB_LEARNING_RATE,
                "max_depth": HGB_MAX_DEPTH,
                "max_features": HGB_MAX_FEATURES,
                "max_iter": HGB_MAX_ITER,
                "max_leaf_nodes": HGB_MAX_LEAF_NODES,
                "min_samples_leaf": HGB_MIN_SAMPLES_LEAF,
                "random_state_rule": "base_random_state + frozen_fold_index",
                "zero_variance_selector": "VarianceThreshold(threshold=0.0)",
            },
            builder=lambda state: build_hist_gradient_boosting_pipeline(
                random_state=state
            ),
        ),
    )


def _fitted_classifier(estimator: Any) -> Any:
    """Return a fitted classifier from a direct estimator or a pipeline."""
    if isinstance(estimator, Pipeline):
        classifier = estimator.named_steps.get("classifier")
        if classifier is None:
            raise ThresholdSensitivityError(
                "Candidate pipeline is missing the classifier step."
            )
        return classifier

    return estimator


def _positive_class_probabilities(
    estimator: Any,
    validation_features: pd.DataFrame,
) -> np.ndarray:
    """Return class-1 probabilities without assuming the class ordering."""
    classifier = _fitted_classifier(estimator)
    if not hasattr(classifier, "classes_"):
        raise ThresholdSensitivityError(
            "Candidate classifier must be fitted before prediction."
        )

    positive_positions = np.flatnonzero(classifier.classes_ == 1)
    if len(positive_positions) != 1:
        raise ThresholdSensitivityError(
            "Candidate classifier must contain exactly one positive class."
        )

    probabilities = estimator.predict_proba(validation_features)
    scores = probabilities[:, int(positive_positions[0])]
    if len(scores) != len(validation_features) or not np.isfinite(scores).all():
        raise ThresholdSensitivityError(
            "Candidate returned invalid positive probabilities."
        )

    return scores.astype(float)


def _feature_count_after_preprocessing(
    estimator: Any,
    input_feature_count: int,
) -> int:
    """Report the fold-fitted feature count after optional variance filtering."""
    if not isinstance(estimator, Pipeline):
        return int(input_feature_count)

    selector = estimator.named_steps.get("variance_threshold")
    if selector is None:
        return int(input_feature_count)

    support = selector.get_support()
    if support.shape != (input_feature_count,):
        raise ThresholdSensitivityError(
            "Variance selector support conflicts with frozen feature columns."
        )

    retained_count = int(support.sum())
    if retained_count == 0:
        raise ThresholdSensitivityError(
            "Variance selector removed every feature in a sensitivity fold."
        )

    return retained_count


def _fit_candidate(
    candidate: FixedCandidate,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    *,
    random_state: int,
) -> Any:
    """Fit one fixed candidate and fail on logistic convergence warnings."""
    estimator = candidate.builder(int(random_state))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning,
            )
            estimator.fit(training_features, training_labels)
    except ConvergenceWarning as error:
        raise ThresholdSensitivityError(
            f"{candidate.model_id} did not converge in a frozen fold."
        ) from error
    except ValueError as error:
        raise ThresholdSensitivityError(
            f"{candidate.model_id} failed to fit a frozen fold."
        ) from error

    return estimator


def _metric_summary(
    fold_metrics: list[BinaryClassificationMetrics],
) -> dict[str, dict[str, float]]:
    """Summarize fold-level metrics without treating folds as a hypothesis test."""
    if not fold_metrics:
        raise ThresholdSensitivityError("No fold metrics were produced.")

    summary: dict[str, dict[str, float]] = {}
    for metric_name in SUMMARY_METRICS:
        values = np.asarray(
            [getattr(metrics, metric_name) for metrics in fold_metrics],
            dtype=float,
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    return summary


def _run_candidate_for_variant(
    candidate: FixedCandidate,
    dataset: FrozenFoldDataset,
    variant: LabelVariant,
    *,
    random_state: int,
    decision_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one fixed candidate across the inherited five cold-drug folds."""
    labels = dataset.labels[variant.label_column]
    scores = np.full(len(labels), np.nan, dtype=float)
    assignment_counts = np.zeros(len(labels), dtype=int)
    fold_results: list[dict[str, Any]] = []
    fold_metrics: list[BinaryClassificationMetrics] = []

    for fold_index in range(dataset.n_splits):
        validation_positions = np.flatnonzero(dataset.fold_indices == fold_index)
        train_positions = np.flatnonzero(dataset.fold_indices != fold_index)
        if not len(validation_positions) or not len(train_positions):
            raise ThresholdSensitivityError(
                f"Frozen fold {fold_index} has an empty partition."
            )

        train_drugs = set(dataset.metadata.iloc[train_positions]["drug_id"])
        validation_drugs = set(
            dataset.metadata.iloc[validation_positions]["drug_id"]
        )
        overlap_count = int(len(train_drugs.intersection(validation_drugs)))
        if overlap_count:
            raise ThresholdSensitivityError(
                f"Frozen fold {fold_index} leaks drug groups."
            )

        training_labels = labels.iloc[train_positions]
        validation_labels = labels.iloc[validation_positions]
        if training_labels.nunique() != 2 or validation_labels.nunique() != 2:
            raise ThresholdSensitivityError(
                f"{variant.label_column} lacks both classes in frozen fold "
                f"{fold_index}."
            )

        fit_random_state = int(random_state) + fold_index
        estimator = _fit_candidate(
            candidate,
            dataset.features.iloc[train_positions],
            training_labels,
            random_state=fit_random_state,
        )
        positive_scores = _positive_class_probabilities(
            estimator,
            dataset.features.iloc[validation_positions],
        )
        metrics = evaluate_binary_classification(
            validation_labels,
            positive_scores,
            decision_threshold=decision_threshold,
        )

        scores[validation_positions] = positive_scores
        assignment_counts[validation_positions] += 1
        fold_metrics.append(metrics)
        fold_results.append(
            {
                "fold_index": fold_index,
                "fit_random_state": fit_random_state,
                "train_pair_count": int(len(train_positions)),
                "validation_pair_count": int(len(validation_positions)),
                "train_drug_count": int(len(train_drugs)),
                "validation_drug_count": int(len(validation_drugs)),
                "drug_overlap_count": overlap_count,
                "train_positive_rate": float(training_labels.mean()),
                "validation_positive_rate": float(validation_labels.mean()),
                "feature_count_after_preprocessing": (
                    _feature_count_after_preprocessing(
                        estimator,
                        len(dataset.feature_columns),
                    )
                ),
                "metrics": metrics.to_dict(),
            }
        )

    if not np.all(assignment_counts == 1) or not np.isfinite(scores).all():
        raise ThresholdSensitivityError(
            f"{candidate.model_id} did not produce one valid OOF score per row."
        )

    pooled_metrics = evaluate_binary_classification(
        labels,
        scores,
        decision_threshold=decision_threshold,
    )
    oof_predictions = pd.DataFrame(
        {
            "threshold_variant": variant.variant_id,
            "label_column": variant.label_column,
            "model_id": candidate.model_id,
            "model_name": candidate.model_name,
            "fold_index": dataset.fold_indices,
            "fit_random_state": (
                int(random_state) + dataset.fold_indices.astype(int)
            ),
            "observed_pair_index": dataset.metadata[
                "observed_pair_index"
            ].to_numpy(),
            "drug_id": dataset.metadata["drug_id"].to_numpy(),
            "target_id": dataset.metadata["target_id"].to_numpy(),
            "y_true": labels.to_numpy(dtype=np.int8),
            "positive_probability": scores,
        }
    ).loc[:, list(SENSITIVITY_OOF_COLUMNS)]

    result = {
        "model_id": candidate.model_id,
        "model_name": candidate.model_name,
        "model_parameters": candidate.model_parameters,
        "fold_results": fold_results,
        "fold_metric_summary": _metric_summary(fold_metrics),
        "pooled_oof_metrics_descriptive_only": pooled_metrics.to_dict(),
        "oof_prediction_count": int(len(oof_predictions)),
    }
    return result, oof_predictions


def run_threshold_sensitivity(
    feature_table: pd.DataFrame,
    reference_oof_assignments: pd.DataFrame,
    frozen_inner_cv_summary: dict[str, Any],
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> ThresholdSensitivityRun:
    """Run both threshold definitions on fixed primary inner-CV folds only."""
    if not 0.0 < float(decision_threshold) < 1.0:
        raise ThresholdSensitivityError(
            "decision_threshold must be strictly between zero and one."
        )

    dataset = build_frozen_fold_dataset(
        feature_table,
        reference_oof_assignments,
        frozen_inner_cv_summary,
        n_splits=n_splits,
        random_state=random_state,
    )

    candidate_models = _candidate_models()
    if not candidate_models:
        raise ThresholdSensitivityError("No fixed candidate models are defined.")

    variant_reports: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    for variant in LABEL_VARIANTS:
        label_values = dataset.labels[variant.label_column]
        model_results: list[dict[str, Any]] = []
        for candidate in candidate_models:
            result, predictions = _run_candidate_for_variant(
                candidate,
                dataset,
                variant,
                random_state=random_state,
                decision_threshold=float(decision_threshold),
            )
            model_results.append(result)
            oof_frames.append(predictions)

        variant_reports.append(
            {
                "variant_id": variant.variant_id,
                "label_column": variant.label_column,
                "kd_threshold_nM": variant.kd_threshold_nM,
                "pKd_threshold": variant.pKd_threshold,
                "positive_count": int(label_values.sum()),
                "negative_count": int((1 - label_values).sum()),
                "positive_rate": float(label_values.mean()),
                "model_results": model_results,
            }
        )

    oof_predictions = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["threshold_variant", "model_id", "observed_pair_index"],
        kind="stable",
    ).reset_index(drop=True)

    expected_prediction_count = (
        len(dataset.features) * len(LABEL_VARIANTS) * len(candidate_models)
    )
    if len(oof_predictions) != expected_prediction_count:
        raise ThresholdSensitivityError(
            "Sensitivity OOF prediction coverage is incomplete."
        )

    report = {
        "analysis_scope": (
            "frozen_outer_training_partition_fixed_inner_cold_drug_folds"
        ),
        "outer_policy": "cold_drug",
        "outer_test_partition_used": False,
        "outer_test_outcomes_selected": False,
        "frozen_fold_source": {
            "source_model_id": REFERENCE_MODEL_ID,
            "source_label_column": PRIMARY_LABEL_COLUMN,
            "source_description": (
                "Fold assignments are recovered from the existing primary "
                "inner-CV OOF structure; prior y_true and probability columns "
                "are not read."
            ),
        },
        "model_selection": {
            "primary_selected_model_id": PRIMARY_SELECTED_MODEL_ID,
            "selection_metric": PRIMARY_SELECTION_METRIC,
            "selection_reopened": False,
            "hyperparameter_tuning_performed": False,
            "interpretation": (
                "Threshold results are descriptive robustness evidence, not a "
                "new model ranking or selection decision."
            ),
        },
        "input_pair_count": int(len(dataset.features)),
        "input_drug_count": int(dataset.metadata["drug_id"].nunique()),
        "input_target_count": int(dataset.metadata["target_id"].nunique()),
        "input_feature_count": int(len(dataset.feature_columns)),
        "n_splits": dataset.n_splits,
        "random_state": int(random_state),
        "decision_threshold": float(decision_threshold),
        "variants": variant_reports,
        "interpretation_limits": [
            "Average precision is the principal imbalance-aware metric; ROC-AUC "
            "is secondary and accuracy is supplementary.",
            "Fold means and pooled OOF values are descriptive; five folds do "
            "not constitute a statistical superiority test.",
            "Changing the affinity threshold changes the prediction task and "
            "class prevalence; results across thresholds are not directly "
            "interchangeable.",
            "This sensitivity analysis does not establish biological validity, "
            "clinical utility, mechanism, or causal effects.",
            "The frozen outer cold-drug holdout is not used for this analysis "
            "and must remain unavailable for subsequent model development.",
        ],
    }

    return ThresholdSensitivityRun(
        report=report,
        oof_predictions=oof_predictions,
    )


def write_threshold_sensitivity_summary(
    run: ThresholdSensitivityRun,
    output_path: str | Path,
) -> Path:
    """Write compact version-controlled threshold-sensitivity evidence."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_threshold_sensitivity_oof_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write detailed local OOF results under an explicit column contract."""
    if tuple(predictions.columns) != SENSITIVITY_OOF_COLUMNS:
        raise ThresholdSensitivityError(
            "Sensitivity OOF columns do not match the frozen contract."
        )
    if predictions.empty:
        raise ThresholdSensitivityError("Sensitivity OOF prediction table is empty.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False, float_format="%.17g")
    return destination


def _read_reference_oof_structure(path: Path) -> pd.DataFrame:
    """Read only structural OOF columns, deliberately excluding outcomes/scores."""
    columns = [
        "model_id",
        "fold_index",
        "observed_pair_index",
        "drug_id",
        "target_id",
    ]
    return pd.read_csv(
        path,
        usecols=columns,
        dtype={"model_id": str, "drug_id": str, "target_id": str},
    )


def _read_feature_table(path: Path) -> pd.DataFrame:
    """Read the frozen transparent features and two pre-specified label columns."""
    columns = [
        "observed_pair_index",
        "drug_id",
        "target_id",
        PRIMARY_LABEL_COLUMN,
        SENSITIVITY_LABEL_COLUMN,
        *FEATURE_COLUMNS,
    ]
    return pd.read_csv(
        path,
        usecols=columns,
        dtype={"drug_id": str, "target_id": str},
    )


def main(argv: list[str] | None = None) -> int:
    """Run the pre-specified 100 nM sensitivity on inherited inner-CV folds."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare pre-specified Davis affinity-label thresholds on frozen "
            "outer-training cold-drug folds without reopening model selection."
        )
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("data/processed/davis_pair_features.csv"),
        help="Local transparent Davis feature table containing both labels.",
    )
    parser.add_argument(
        "--inner-cv-summary",
        type=Path,
        default=Path("reports/davis_inner_cold_drug_cv.json"),
        help="Committed primary inner-CV report defining the frozen contract.",
    )
    parser.add_argument(
        "--inner-oof-predictions",
        type=Path,
        default=Path("data/interim/davis_inner_cold_drug_oof_predictions.csv"),
        help="Ignored primary OOF file used only to recover fixed fold membership.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of inherited cold-drug inner folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Frozen base random state; fold fits use base plus fold index.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=DEFAULT_DECISION_THRESHOLD,
        help="Fixed probability threshold for confusion-matrix metrics.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_threshold_sensitivity.json"),
        help="Version-controlled JSON summary destination.",
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path("data/interim/davis_threshold_sensitivity_oof_predictions.csv"),
        help="Ignored detailed local OOF-prediction CSV destination.",
    )
    args = parser.parse_args(argv)

    try:
        summary = load_inner_cv_summary(args.inner_cv_summary)
        reference_oof = _read_reference_oof_structure(
            args.inner_oof_predictions
        )
        feature_table = _read_feature_table(args.feature_table)
        run = run_threshold_sensitivity(
            feature_table,
            reference_oof,
            summary,
            n_splits=args.n_splits,
            random_state=args.random_state,
            decision_threshold=args.decision_threshold,
        )
        summary_path = write_threshold_sensitivity_summary(
            run,
            args.summary_output,
        )
        oof_path = write_threshold_sensitivity_oof_predictions(
            run.oof_predictions,
            args.oof_output,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        print(f"Threshold sensitivity failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(run.report, indent=2, sort_keys=True))
    print(f"Threshold-sensitivity summary written to: {summary_path}")
    print(f"Threshold-sensitivity OOF predictions written to: {oof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
