"""Run a leakage-safe histogram gradient-boosting Davis DTI model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from src.models.dataset import ModelDataset, POLICIES, load_train_test_data
from src.models.evaluation import (
    BinaryClassificationMetrics,
    DEFAULT_DECISION_THRESHOLD,
    evaluate_binary_classification,
)


DEFAULT_RANDOM_STATE = 20260830
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_ITER = 200
DEFAULT_MAX_LEAF_NODES = 15
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_SAMPLES_LEAF = 20
DEFAULT_L2_REGULARIZATION = 1.0
DEFAULT_MAX_FEATURES = 0.8
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_EARLY_STOPPING = False
DEFAULT_ZERO_VARIANCE_THRESHOLD = 0.0


class HistGradientBoostingError(ValueError):
    """Raised when the fixed gradient-boosting protocol cannot be followed."""


@dataclass(frozen=True)
class HistGradientBoostingResult:
    """Serializable evidence from one fixed gradient-boosting evaluation."""

    policy: str
    label_column: str
    input_feature_count: int
    retained_feature_count: int
    retained_feature_columns: tuple[str, ...]
    removed_zero_variance_features: tuple[str, ...]
    training_positive_rate: float
    fitted_iteration_count: int
    trees_per_iteration: int
    internal_early_stopping_used: bool
    metrics: BinaryClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible result record."""
        return {
            "model_name": "HistGradientBoostingClassifier",
            "model_parameters": {
                "class_weight": DEFAULT_CLASS_WEIGHT,
                "early_stopping": DEFAULT_EARLY_STOPPING,
                "l2_regularization": DEFAULT_L2_REGULARIZATION,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "loss": "log_loss",
                "max_depth": DEFAULT_MAX_DEPTH,
                "max_features": DEFAULT_MAX_FEATURES,
                "max_leaf_nodes": DEFAULT_MAX_LEAF_NODES,
                "max_iter": DEFAULT_MAX_ITER,
                "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF,
                "random_state": DEFAULT_RANDOM_STATE,
                "zero_variance_selector": (
                    "VarianceThreshold(threshold=0.0)"
                ),
            },
            "policy": self.policy,
            "label_column": self.label_column,
            "input_feature_count": self.input_feature_count,
            "retained_feature_count": self.retained_feature_count,
            "retained_feature_columns": self.retained_feature_columns,
            "removed_zero_variance_features": (
                self.removed_zero_variance_features
            ),
            "training_positive_rate": self.training_positive_rate,
            "fitted_iteration_count": self.fitted_iteration_count,
            "trees_per_iteration": self.trees_per_iteration,
            "internal_early_stopping_used": (
                self.internal_early_stopping_used
            ),
            "metrics": self.metrics.to_dict(),
        }


def build_hist_gradient_boosting_pipeline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build an unfitted selector plus fixed regularized boosting model."""
    return Pipeline(
        steps=[
            (
                "variance_threshold",
                VarianceThreshold(
                    threshold=DEFAULT_ZERO_VARIANCE_THRESHOLD
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    categorical_features=None,
                    class_weight=DEFAULT_CLASS_WEIGHT,
                    early_stopping=DEFAULT_EARLY_STOPPING,
                    l2_regularization=DEFAULT_L2_REGULARIZATION,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    loss="log_loss",
                    max_depth=DEFAULT_MAX_DEPTH,
                    max_features=DEFAULT_MAX_FEATURES,
                    max_leaf_nodes=DEFAULT_MAX_LEAF_NODES,
                    max_iter=DEFAULT_MAX_ITER,
                    min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def _selected_feature_columns(
    pipeline: Pipeline,
    feature_columns: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Map the fitted variance selector back to original feature names."""
    selector = pipeline.named_steps["variance_threshold"]
    support = selector.get_support()

    if support.shape != (len(feature_columns),):
        raise HistGradientBoostingError(
            "Variance selector support does not match input features."
        )

    retained = tuple(
        column
        for column, selected in zip(
            feature_columns,
            support,
            strict=True,
        )
        if bool(selected)
    )

    removed = tuple(
        column
        for column, selected in zip(
            feature_columns,
            support,
            strict=True,
        )
        if not bool(selected)
    )

    if not retained:
        raise HistGradientBoostingError(
            "Variance selector removed every feature from training data."
        )

    return retained, removed


def _positive_class_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return positive-class probabilities without assuming class order."""
    classifier = pipeline.named_steps["classifier"]
    positive_positions = np.flatnonzero(classifier.classes_ == 1)

    if len(positive_positions) != 1:
        raise HistGradientBoostingError(
            "Fitted classifier must contain exactly one positive class."
        )

    probabilities = pipeline.predict_proba(features)
    positive_scores = probabilities[:, int(positive_positions[0])]

    if len(positive_scores) != len(features):
        raise HistGradientBoostingError(
            "Classifier returned an unexpected number of probabilities."
        )

    if not np.isfinite(positive_scores).all():
        raise HistGradientBoostingError(
            "Classifier returned non-finite positive probabilities."
        )

    return positive_scores.astype(float)


def _fit_summary(pipeline: Pipeline) -> tuple[int, int, bool]:
    """Return fitted iteration evidence and enforce no internal early stopping."""
    classifier = pipeline.named_steps["classifier"]

    internal_early_stopping_used = bool(classifier.do_early_stopping_)

    if internal_early_stopping_used:
        raise HistGradientBoostingError(
            "Internal early stopping was enabled despite the fixed protocol."
        )

    fitted_iteration_count = int(classifier.n_iter_)

    if fitted_iteration_count != DEFAULT_MAX_ITER:
        raise HistGradientBoostingError(
            "Boosting iteration count does not match the fixed protocol."
        )

    trees_per_iteration = int(classifier.n_trees_per_iteration_)

    if trees_per_iteration != 1:
        raise HistGradientBoostingError(
            "Binary gradient boosting must fit one tree per iteration."
        )

    return (
        fitted_iteration_count,
        trees_per_iteration,
        internal_early_stopping_used,
    )


def run_hist_gradient_boosting_experiment(
    dataset: ModelDataset,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> HistGradientBoostingResult:
    """Fit on frozen training data and evaluate once on frozen test data."""
    pipeline = build_hist_gradient_boosting_pipeline(
        random_state=random_state
    )

    pipeline.fit(dataset.X_train, dataset.y_train)

    retained_features, removed_features = _selected_feature_columns(
        pipeline,
        dataset.feature_columns,
    )

    positive_scores = _positive_class_probabilities(
        pipeline,
        dataset.X_test,
    )

    metrics = evaluate_binary_classification(
        dataset.y_test,
        positive_scores,
        decision_threshold=decision_threshold,
    )

    (
        fitted_iteration_count,
        trees_per_iteration,
        internal_early_stopping_used,
    ) = _fit_summary(pipeline)

    return HistGradientBoostingResult(
        policy=dataset.policy,
        label_column=dataset.label_column,
        input_feature_count=len(dataset.feature_columns),
        retained_feature_count=len(retained_features),
        retained_feature_columns=retained_features,
        removed_zero_variance_features=removed_features,
        training_positive_rate=float(dataset.y_train.mean()),
        fitted_iteration_count=fitted_iteration_count,
        trees_per_iteration=trees_per_iteration,
        internal_early_stopping_used=internal_early_stopping_used,
        metrics=metrics,
    )


def write_hist_gradient_boosting_result(
    result: HistGradientBoostingResult,
    output_path: str | Path,
) -> Path:
    """Write compact model evidence without serializing fitted estimators."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return destination


def main(argv: list[str] | None = None) -> int:
    """Run fixed histogram gradient boosting from local Davis artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fixed histogram gradient-boosting Davis DTI model."
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
        help="Pre-specified binary label column to evaluate.",
    )

    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="cold_drug",
        help="Frozen split policy to evaluate.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Fixed random state for reproducible histogram binning.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/davis_hist_gradient_boosting_cold_drug.json"),
        help="Version-controlled JSON result destination.",
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

        result = run_hist_gradient_boosting_experiment(
            dataset,
            random_state=args.random_state,
        )

        output_path = write_hist_gradient_boosting_result(
            result,
            args.output,
        )

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(
            f"Histogram gradient-boosting evaluation failed: {error}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"Gradient-boosting result written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())