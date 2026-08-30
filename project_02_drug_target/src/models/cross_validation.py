"""Run inner, drug-grouped cross-validation without using outer test data."""

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.models.baselines import DUMMY_STRATEGY
from src.models.dataset import ModelDataset, load_train_test_data
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
PRIMARY_COMPARISON_METRIC = "average_precision"

DUMMY_MODEL_ID = "dummy_prior"
LOGISTIC_MODEL_ID = "logistic_regression_balanced"
RANDOM_FOREST_MODEL_ID = "random_forest_balanced"
HIST_GRADIENT_BOOSTING_MODEL_ID = "hist_gradient_boosting_balanced"

CANDIDATE_MODEL_IDS = (
    DUMMY_MODEL_ID,
    LOGISTIC_MODEL_ID,
    RANDOM_FOREST_MODEL_ID,
    HIST_GRADIENT_BOOSTING_MODEL_ID,
)

SUMMARY_METRICS = (
    "average_precision",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
)

OOF_COLUMNS = (
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


class CrossValidationError(ValueError):
    """Raised when inner drug-grouped cross-validation is invalid."""


@dataclass(frozen=True)
class ModelCandidate:
    """One fixed model candidate used in every inner CV fold."""

    model_id: str
    model_name: str
    model_parameters: dict[str, Any]
    builder: Callable[[int], Any]


@dataclass(frozen=True)
class FoldMetricSummary:
    """Mean and variability of one metric across inner validation folds."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible aggregate statistics."""
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class FoldResult:
    """One candidate's fit and validation evidence for one group-CV fold."""

    fold_index: int
    fit_random_state: int
    train_pair_count: int
    validation_pair_count: int
    train_drug_count: int
    validation_drug_count: int
    drug_overlap_count: int
    train_positive_rate: float
    validation_positive_rate: float
    feature_count_after_preprocessing: int
    metrics: BinaryClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fold evidence."""
        return {
            "fold_index": self.fold_index,
            "fit_random_state": self.fit_random_state,
            "train_pair_count": self.train_pair_count,
            "validation_pair_count": self.validation_pair_count,
            "train_drug_count": self.train_drug_count,
            "validation_drug_count": self.validation_drug_count,
            "drug_overlap_count": self.drug_overlap_count,
            "train_positive_rate": self.train_positive_rate,
            "validation_positive_rate": self.validation_positive_rate,
            "feature_count_after_preprocessing": (
                self.feature_count_after_preprocessing
            ),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class CandidateCVResult:
    """All inner-fold and pooled evidence for one fixed candidate model."""

    model_id: str
    model_name: str
    model_parameters: dict[str, Any]
    fold_results: tuple[FoldResult, ...]
    fold_metric_summary: dict[str, FoldMetricSummary]
    pooled_oof_metrics: BinaryClassificationMetrics
    oof_prediction_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible candidate evidence."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_parameters": self.model_parameters,
            "fold_results": [
                fold_result.to_dict()
                for fold_result in self.fold_results
            ],
            "fold_metric_summary": {
                metric_name: summary.to_dict()
                for metric_name, summary in self.fold_metric_summary.items()
            },
            "pooled_oof_metrics": self.pooled_oof_metrics.to_dict(),
            "oof_prediction_count": self.oof_prediction_count,
        }


@dataclass(frozen=True)
class InnerCVSummary:
    """Version-controlled summary of inner cold-drug cross-validation."""

    outer_policy: str
    cv_scope: str
    outer_test_partition_used: bool
    group_column: str
    input_pair_count: int
    input_drug_count: int
    input_feature_count: int
    label_column: str
    n_splits: int
    shuffle: bool
    random_state: int
    fit_random_state_rule: str
    primary_comparison_metric: str
    model_results: tuple[CandidateCVResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible inner-CV evidence."""
        return {
            "outer_policy": self.outer_policy,
            "cv_scope": self.cv_scope,
            "outer_test_partition_used": self.outer_test_partition_used,
            "group_column": self.group_column,
            "input_pair_count": self.input_pair_count,
            "input_drug_count": self.input_drug_count,
            "input_feature_count": self.input_feature_count,
            "label_column": self.label_column,
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "fit_random_state_rule": self.fit_random_state_rule,
            "primary_comparison_metric": self.primary_comparison_metric,
            "model_results": [
                model_result.to_dict()
                for model_result in self.model_results
            ],
        }


@dataclass(frozen=True)
class InnerCVRun:
    """Inner-CV summary plus local out-of-fold predictions."""

    summary: InnerCVSummary
    oof_predictions: pd.DataFrame


def _build_dummy_prior(random_state: int) -> DummyClassifier:
    """Build the fixed empirical-prior baseline."""
    return DummyClassifier(
        strategy=DUMMY_STRATEGY,
        random_state=int(random_state),
    )


def _candidate_models() -> tuple[ModelCandidate, ...]:
    """Return the fixed candidate models in reporting order."""
    return (
        ModelCandidate(
            model_id=DUMMY_MODEL_ID,
            model_name="DummyClassifier",
            model_parameters={
                "random_state_rule": "base_random_state + fold_index",
                "strategy": DUMMY_STRATEGY,
            },
            builder=_build_dummy_prior,
        ),
        ModelCandidate(
            model_id=LOGISTIC_MODEL_ID,
            model_name="LogisticRegression",
            model_parameters={
                "C": LOGISTIC_C,
                "class_weight": "balanced",
                "l1_ratio": LOGISTIC_L1_RATIO,
                "max_iter": LOGISTIC_MAX_ITER,
                "random_state_rule": "base_random_state + fold_index",
                "regularization": "L2",
                "solver": LOGISTIC_SOLVER,
                "standardization": "StandardScaler",
            },
            builder=lambda state: build_logistic_pipeline(
                "balanced",
                random_state=state,
            ),
        ),
        ModelCandidate(
            model_id=RANDOM_FOREST_MODEL_ID,
            model_name="RandomForestClassifier",
            model_parameters={
                "class_weight": RF_CLASS_WEIGHT,
                "max_depth": RF_MAX_DEPTH,
                "max_features": RF_MAX_FEATURES,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                "n_estimators": RF_N_ESTIMATORS,
                "n_jobs": RF_N_JOBS,
                "random_state_rule": "base_random_state + fold_index",
                "zero_variance_selector": (
                    "VarianceThreshold(threshold=0.0)"
                ),
            },
            builder=lambda state: build_random_forest_pipeline(
                random_state=state
            ),
        ),
        ModelCandidate(
            model_id=HIST_GRADIENT_BOOSTING_MODEL_ID,
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
                "random_state_rule": "base_random_state + fold_index",
                "zero_variance_selector": (
                    "VarianceThreshold(threshold=0.0)"
                ),
            },
            builder=lambda state: build_hist_gradient_boosting_pipeline(
                random_state=state
            ),
        ),
    )


def _validated_n_splits(value: int, unique_drug_count: int) -> int:
    """Validate the requested number of group-CV folds."""
    try:
        n_splits = int(value)
    except (TypeError, ValueError) as error:
        raise CrossValidationError("n_splits must be an integer.") from error

    if n_splits < 2:
        raise CrossValidationError("n_splits must be at least two.")

    if n_splits > unique_drug_count:
        raise CrossValidationError(
            "n_splits cannot exceed the number of unique training drugs."
        )

    return n_splits


def _validated_inner_inputs(
    dataset: ModelDataset,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray, int]:
    """Return validated outer-training data without reading outer test values."""
    if dataset.policy != "cold_drug":
        raise CrossValidationError(
            "Inner grouped CV is defined only for the cold_drug policy."
        )

    if dataset.X_train.empty:
        raise CrossValidationError("Outer-training feature matrix is empty.")

    if len(dataset.X_train) != len(dataset.y_train):
        raise CrossValidationError(
            "Outer-training feature and label lengths do not match."
        )

    if len(dataset.X_train) != len(dataset.train_metadata):
        raise CrossValidationError(
            "Outer-training feature and metadata lengths do not match."
        )

    if tuple(dataset.X_train.columns) != tuple(dataset.feature_columns):
        raise CrossValidationError(
            "Outer-training features do not match the frozen feature contract."
        )

    required_metadata_columns = {
        "observed_pair_index",
        "drug_id",
        "target_id",
    }

    missing_metadata_columns = required_metadata_columns.difference(
        dataset.train_metadata.columns
    )

    if missing_metadata_columns:
        raise CrossValidationError(
            "Training metadata is missing columns: "
            f"{sorted(missing_metadata_columns)}"
        )

    features = dataset.X_train.reset_index(drop=True).copy()
    labels = dataset.y_train.astype("int8").reset_index(drop=True).copy()

    if labels.nunique() != 2:
        raise CrossValidationError(
            "Outer-training labels must contain both classes."
        )

    metadata = dataset.train_metadata.loc[
        :,
        ["observed_pair_index", "drug_id", "target_id"],
    ].reset_index(drop=True).copy()

    if metadata["observed_pair_index"].duplicated().any():
        raise CrossValidationError(
            "Training metadata contains duplicate observed_pair_index values."
        )

    if metadata["drug_id"].isna().any():
        raise CrossValidationError(
            "Training metadata contains missing drug identifiers."
        )

    groups = metadata["drug_id"].astype(str).str.strip().to_numpy()

    if np.any(groups == ""):
        raise CrossValidationError(
            "Training metadata contains empty drug identifiers."
        )

    metadata["drug_id"] = groups

    unique_drug_count = int(pd.Series(groups).nunique())
    checked_n_splits = _validated_n_splits(
        n_splits,
        unique_drug_count,
    )

    return features, labels, metadata, groups, checked_n_splits


def _fold_positions(
    labels: pd.Series,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Create reproducible inner folds from labels and drug groups only."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )

    split_features = np.zeros((len(labels), 1), dtype=np.int8)

    try:
        folds = tuple(
            splitter.split(
                split_features,
                labels.to_numpy(),
                groups=groups,
            )
        )
    except ValueError as error:
        raise CrossValidationError(
            "Could not create stratified drug-group folds."
        ) from error

    if len(folds) != n_splits:
        raise CrossValidationError(
            "Splitter returned an unexpected number of folds."
        )

    return folds


def _fitted_classifier(estimator: Any) -> Any:
    """Return a fitted classifier from a direct model or a pipeline."""
    if isinstance(estimator, Pipeline):
        classifier = estimator.named_steps.get("classifier")

        if classifier is None:
            raise CrossValidationError(
                "Pipeline is missing its classifier step."
            )

        return classifier

    return estimator


def _positive_class_probabilities(
    estimator: Any,
    validation_features: pd.DataFrame,
) -> np.ndarray:
    """Return class-1 probabilities without assuming class ordering."""
    classifier = _fitted_classifier(estimator)

    if not hasattr(classifier, "classes_"):
        raise CrossValidationError(
            "Candidate classifier must be fitted before prediction."
        )

    positive_positions = np.flatnonzero(classifier.classes_ == 1)

    if len(positive_positions) != 1:
        raise CrossValidationError(
            "Candidate classifier must contain exactly one positive class."
        )

    probabilities = estimator.predict_proba(validation_features)
    positive_scores = probabilities[:, int(positive_positions[0])]

    if len(positive_scores) != len(validation_features):
        raise CrossValidationError(
            "Candidate returned an unexpected number of probabilities."
        )

    if not np.isfinite(positive_scores).all():
        raise CrossValidationError(
            "Candidate returned non-finite positive probabilities."
        )

    return positive_scores.astype(float)


def _feature_count_after_preprocessing(
    estimator: Any,
    input_feature_count: int,
) -> int:
    """Return the fold-fitted feature count after optional variance filtering."""
    if not isinstance(estimator, Pipeline):
        return int(input_feature_count)

    selector = estimator.named_steps.get("variance_threshold")

    if selector is None:
        return int(input_feature_count)

    support = selector.get_support()

    if support.shape != (input_feature_count,):
        raise CrossValidationError(
            "Variance selector support does not match fold input features."
        )

    selected_feature_count = int(support.sum())

    if selected_feature_count == 0:
        raise CrossValidationError(
            "Variance selector removed every feature in an inner fold."
        )

    return selected_feature_count


def _fit_candidate(
    candidate: ModelCandidate,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    *,
    random_state: int,
) -> Any:
    """Fit one candidate while treating logistic convergence warnings as errors."""
    estimator = candidate.builder(int(random_state))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning,
            )
            estimator.fit(training_features, training_labels)
    except ConvergenceWarning as error:
        raise CrossValidationError(
            f"{candidate.model_id} did not converge in an inner CV fold."
        ) from error
    except ValueError as error:
        raise CrossValidationError(
            f"{candidate.model_id} failed to fit an inner CV fold."
        ) from error

    return estimator


def _evaluate_fold(
    candidate: ModelCandidate,
    features: pd.DataFrame,
    labels: pd.Series,
    groups: np.ndarray,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    *,
    fold_index: int,
    fit_random_state: int,
    decision_threshold: float,
) -> tuple[FoldResult, np.ndarray]:
    """Fit one candidate on an inner train fold and score its validation fold."""
    train_drugs = set(groups[train_positions])
    validation_drugs = set(groups[validation_positions])
    drug_overlap_count = int(len(train_drugs.intersection(validation_drugs)))

    if drug_overlap_count:
        raise CrossValidationError(
            f"Fold {fold_index} contains overlapping training and "
            "validation drugs."
        )

    training_labels = labels.iloc[train_positions]
    validation_labels = labels.iloc[validation_positions]

    if training_labels.nunique() != 2:
        raise CrossValidationError(
            f"Fold {fold_index} training labels do not contain both classes."
        )

    if validation_labels.nunique() != 2:
        raise CrossValidationError(
            f"Fold {fold_index} validation labels do not contain both classes."
        )

    estimator = _fit_candidate(
        candidate,
        features.iloc[train_positions],
        training_labels,
        random_state=fit_random_state,
    )

    positive_scores = _positive_class_probabilities(
        estimator,
        features.iloc[validation_positions],
    )

    metrics = evaluate_binary_classification(
        validation_labels,
        positive_scores,
        decision_threshold=decision_threshold,
    )

    fold_result = FoldResult(
        fold_index=int(fold_index),
        fit_random_state=int(fit_random_state),
        train_pair_count=int(len(train_positions)),
        validation_pair_count=int(len(validation_positions)),
        train_drug_count=int(len(train_drugs)),
        validation_drug_count=int(len(validation_drugs)),
        drug_overlap_count=drug_overlap_count,
        train_positive_rate=float(training_labels.mean()),
        validation_positive_rate=float(validation_labels.mean()),
        feature_count_after_preprocessing=_feature_count_after_preprocessing(
            estimator,
            len(features.columns),
        ),
        metrics=metrics,
    )

    return fold_result, positive_scores


def _fold_metric_summary(
    fold_results: tuple[FoldResult, ...],
) -> dict[str, FoldMetricSummary]:
    """Summarize candidate metrics across inner validation folds."""
    summaries: dict[str, FoldMetricSummary] = {}

    for metric_name in SUMMARY_METRICS:
        values = np.asarray(
            [
                getattr(fold_result.metrics, metric_name)
                for fold_result in fold_results
            ],
            dtype=float,
        )

        summaries[metric_name] = FoldMetricSummary(
            mean=float(values.mean()),
            standard_deviation=float(values.std(ddof=1)),
            minimum=float(values.min()),
            maximum=float(values.max()),
        )

    return summaries


def _candidate_oof_frame(
    candidate: ModelCandidate,
    metadata: pd.DataFrame,
    oof_fold_indices: np.ndarray,
    oof_fit_random_states: np.ndarray,
    labels: pd.Series,
    oof_scores: np.ndarray,
) -> pd.DataFrame:
    """Build one traceable local OOF prediction table for a candidate."""
    frame = pd.DataFrame(
        {
            "model_id": candidate.model_id,
            "model_name": candidate.model_name,
            "fold_index": oof_fold_indices.astype(int),
            "fit_random_state": oof_fit_random_states.astype(int),
            "observed_pair_index": metadata[
                "observed_pair_index"
            ].to_numpy(),
            "drug_id": metadata["drug_id"].to_numpy(),
            "target_id": metadata["target_id"].to_numpy(),
            "y_true": labels.to_numpy(dtype=np.int8),
            "positive_probability": oof_scores.astype(float),
        }
    )

    return frame.loc[:, list(OOF_COLUMNS)]


def run_inner_cold_drug_cv(
    dataset: ModelDataset,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> InnerCVRun:
    """Compare all fixed candidates within the outer-training drugs only."""
    (
        features,
        labels,
        metadata,
        groups,
        checked_n_splits,
    ) = _validated_inner_inputs(dataset, n_splits)

    fold_positions = _fold_positions(
        labels,
        groups,
        n_splits=checked_n_splits,
        random_state=random_state,
    )

    candidate_results: list[CandidateCVResult] = []
    oof_frames: list[pd.DataFrame] = []

    for candidate in _candidate_models():
        oof_scores = np.full(len(labels), np.nan, dtype=float)
        oof_fold_indices = np.full(len(labels), -1, dtype=int)
        oof_fit_random_states = np.full(len(labels), -1, dtype=int)
        oof_assignment_counts = np.zeros(len(labels), dtype=int)
        fold_results: list[FoldResult] = []

        for fold_index, (
            train_positions,
            validation_positions,
        ) in enumerate(fold_positions):
            fit_random_state = int(random_state) + fold_index

            fold_result, positive_scores = _evaluate_fold(
                candidate,
                features,
                labels,
                groups,
                train_positions,
                validation_positions,
                fold_index=fold_index,
                fit_random_state=fit_random_state,
                decision_threshold=decision_threshold,
            )

            oof_scores[validation_positions] = positive_scores
            oof_fold_indices[validation_positions] = fold_index
            oof_fit_random_states[validation_positions] = (
                fit_random_state
            )
            oof_assignment_counts[validation_positions] += 1
            fold_results.append(fold_result)

        if not np.all(oof_assignment_counts == 1):
            raise CrossValidationError(
                f"{candidate.model_id} did not produce exactly one "
                "out-of-fold prediction for every training pair."
            )

        if np.any(oof_fold_indices < 0):
            raise CrossValidationError(
                f"{candidate.model_id} has unassigned inner-fold indices."
            )

        if np.any(oof_fit_random_states < 0):
            raise CrossValidationError(
                f"{candidate.model_id} has unassigned fit random states."
            )

        if not np.isfinite(oof_scores).all():
            raise CrossValidationError(
                f"{candidate.model_id} has non-finite OOF probabilities."
            )

        folded_results = tuple(fold_results)

        pooled_oof_metrics = evaluate_binary_classification(
            labels,
            oof_scores,
            decision_threshold=decision_threshold,
        )

        candidate_results.append(
            CandidateCVResult(
                model_id=candidate.model_id,
                model_name=candidate.model_name,
                model_parameters=candidate.model_parameters,
                fold_results=folded_results,
                fold_metric_summary=_fold_metric_summary(folded_results),
                pooled_oof_metrics=pooled_oof_metrics,
                oof_prediction_count=int(len(oof_scores)),
            )
        )

        oof_frames.append(
            _candidate_oof_frame(
                candidate,
                metadata,
                oof_fold_indices,
                oof_fit_random_states,
                labels,
                oof_scores,
            )
        )

    oof_predictions = pd.concat(
        oof_frames,
        ignore_index=True,
    ).sort_values(
        ["model_id", "observed_pair_index"],
        kind="stable",
    ).reset_index(drop=True)

    if len(oof_predictions) != len(labels) * len(candidate_results):
        raise CrossValidationError(
            "OOF prediction row count does not match candidate coverage."
        )

    summary = InnerCVSummary(
        outer_policy=dataset.policy,
        cv_scope="frozen_outer_training_partition_only",
        outer_test_partition_used=False,
        group_column="drug_id",
        input_pair_count=int(len(features)),
        input_drug_count=int(pd.Series(groups).nunique()),
        input_feature_count=int(len(features.columns)),
        label_column=dataset.label_column,
        n_splits=checked_n_splits,
        shuffle=True,
        random_state=int(random_state),
        fit_random_state_rule="base_random_state + fold_index",
        primary_comparison_metric=PRIMARY_COMPARISON_METRIC,
        model_results=tuple(candidate_results),
    )

    return InnerCVRun(
        summary=summary,
        oof_predictions=oof_predictions,
    )


def write_inner_cv_summary(
    summary: InnerCVSummary,
    output_path: str | Path,
) -> Path:
    """Write compact, version-controlled inner-CV evidence."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return destination


def write_oof_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write local ignored OOF predictions for later error analysis."""
    if tuple(predictions.columns) != OOF_COLUMNS:
        raise CrossValidationError(
            "OOF prediction columns do not match the frozen contract."
        )

    if predictions.empty:
        raise CrossValidationError("OOF prediction table is empty.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        destination,
        index=False,
        float_format="%.17g",
    )

    return destination


def main(argv: list[str] | None = None) -> int:
    """Run inner cold-drug CV from local frozen Davis artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed DTI models by inner drug-grouped CV without using "
            "the outer test partition."
        )
    )

    parser.add_argument(
        "--feature-table",
        type=Path,
        default=Path("data/processed/davis_pair_features.csv"),
        help="Local feature-table CSV from src.features.representations.",
    )

    parser.add_argument(
        "--split-assignments",
        type=Path,
        default=Path("data/interim/davis_split_assignments.csv"),
        help="Local frozen split-assignment CSV from src.data.splits.",
    )

    parser.add_argument(
        "--label-column",
        default="interaction_kd_le_1000_nM",
        help="Pre-specified binary label column.",
    )

    parser.add_argument(
        "--policy",
        choices=("cold_drug",),
        default="cold_drug",
        help="Frozen outer split policy; only cold_drug is valid here.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of inner drug-grouped folds.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Base random state for folds and candidate fits.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/davis_inner_cold_drug_cv.json"),
        help="Version-controlled JSON summary destination.",
    )

    parser.add_argument(
        "--oof-output",
        type=Path,
        default=Path(
            "data/interim/davis_inner_cold_drug_oof_predictions.csv"
        ),
        help="Ignored local OOF predictions for later error analysis.",
    )

    args = parser.parse_args(argv)

    try:
        feature_table = pd.read_csv(
            args.feature_table,
            dtype={"drug_id": str, "target_id": str},
        )

        split_assignments = pd.read_csv(
            args.split_assignments,
            dtype={"split_policy": str, "partition": str},
        )

        dataset = load_train_test_data(
            feature_table,
            split_assignments,
            label_column=args.label_column,
            policy=args.policy,
        )

        run = run_inner_cold_drug_cv(
            dataset,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

        summary_path = write_inner_cv_summary(
            run.summary,
            args.summary_output,
        )

        oof_path = write_oof_predictions(
            run.oof_predictions,
            args.oof_output,
        )

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Inner cross-validation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(run.summary.to_dict(), indent=2, sort_keys=True))
    print(f"Inner-CV summary written to: {summary_path}")
    print(f"OOF predictions written to: {oof_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())