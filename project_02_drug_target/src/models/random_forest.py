"""Run a leakage-safe Random Forest Davis DTI baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from src.models.dataset import ModelDataset, POLICIES, load_train_test_data
from src.models.evaluation import (
    BinaryClassificationMetrics,
    DEFAULT_DECISION_THRESHOLD,
    evaluate_binary_classification,
)


DEFAULT_RANDOM_STATE = 20260830
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH = 12
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_MAX_FEATURES = "sqrt"
DEFAULT_CLASS_WEIGHT = "balanced"
DEFAULT_N_JOBS = 1
DEFAULT_ZERO_VARIANCE_THRESHOLD = 0.0


class RandomForestError(ValueError):
    """Raised when the fixed Random Forest protocol cannot be followed."""


@dataclass(frozen=True)
class FeatureImportance:
    """One retained feature's impurity-based model importance."""

    feature: str
    impurity_importance: float


@dataclass(frozen=True)
class RandomForestResult:
    """Serializable evidence from one fixed Random Forest evaluation."""

    policy: str
    label_column: str
    input_feature_count: int
    retained_feature_count: int
    retained_feature_columns: tuple[str, ...]
    removed_zero_variance_features: tuple[str, ...]
    training_positive_rate: float
    tree_count: int
    maximum_tree_depth: int
    mean_tree_depth: float
    feature_importance_sum: float
    feature_importance_ranking: tuple[FeatureImportance, ...]
    metrics: BinaryClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible result record."""
        return {
            "model_name": "RandomForestClassifier",
            "model_parameters": {
                "bootstrap": True,
                "class_weight": DEFAULT_CLASS_WEIGHT,
                "criterion": "gini",
                "max_depth": DEFAULT_MAX_DEPTH,
                "max_features": DEFAULT_MAX_FEATURES,
                "min_samples_leaf": DEFAULT_MIN_SAMPLES_LEAF,
                "n_estimators": DEFAULT_N_ESTIMATORS,
                "n_jobs": DEFAULT_N_JOBS,
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
            "tree_count": self.tree_count,
            "maximum_tree_depth": self.maximum_tree_depth,
            "mean_tree_depth": self.mean_tree_depth,
            "feature_importance_sum": self.feature_importance_sum,
            "feature_importance_ranking": [
                asdict(importance)
                for importance in self.feature_importance_ranking
            ],
            "metrics": self.metrics.to_dict(),
        }


def build_random_forest_pipeline(
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build an unfitted selector plus fixed conservative Random Forest."""
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
                RandomForestClassifier(
                    bootstrap=True,
                    class_weight=DEFAULT_CLASS_WEIGHT,
                    criterion="gini",
                    max_depth=DEFAULT_MAX_DEPTH,
                    max_features=DEFAULT_MAX_FEATURES,
                    min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
                    n_estimators=DEFAULT_N_ESTIMATORS,
                    n_jobs=DEFAULT_N_JOBS,
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
        raise RandomForestError(
            "Variance selector support does not match the input features."
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
        raise RandomForestError(
            "Variance selector removed every feature from the training data."
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
        raise RandomForestError(
            "Fitted classifier must contain exactly one positive class."
        )

    probabilities = pipeline.predict_proba(features)
    positive_scores = probabilities[:, int(positive_positions[0])]

    if len(positive_scores) != len(features):
        raise RandomForestError(
            "Classifier returned an unexpected number of probabilities."
        )

    if not np.isfinite(positive_scores).all():
        raise RandomForestError(
            "Classifier returned non-finite positive probabilities."
        )

    return positive_scores.astype(float)


def _feature_importance_ranking(
    pipeline: Pipeline,
    retained_feature_columns: tuple[str, ...],
) -> tuple[tuple[FeatureImportance, ...], float]:
    """Return ranked impurity importances aligned to retained features."""
    classifier = pipeline.named_steps["classifier"]
    importances = classifier.feature_importances_

    if importances.shape != (len(retained_feature_columns),):
        raise RandomForestError(
            "Feature-importance shape does not match retained features."
        )

    if not np.isfinite(importances).all() or np.any(importances < 0.0):
        raise RandomForestError(
            "Classifier returned invalid impurity-based feature importances."
        )

    importance_sum = float(importances.sum())

    if not np.isclose(importance_sum, 1.0, rtol=1e-9, atol=1e-12):
        raise RandomForestError(
            "Random Forest feature importances do not sum to one."
        )

    ranking = [
        FeatureImportance(
            feature=column,
            impurity_importance=float(importance),
        )
        for column, importance in zip(
            retained_feature_columns,
            importances,
            strict=True,
        )
    ]

    ranking.sort(
        key=lambda item: (
            -item.impurity_importance,
            item.feature,
        )
    )

    return tuple(ranking), importance_sum


def _tree_depth_summary(
    pipeline: Pipeline,
) -> tuple[int, float, int]:
    """Return fitted-tree count and depth summary for overfitting evidence."""
    classifier = pipeline.named_steps["classifier"]

    depths = np.asarray(
        [estimator.tree_.max_depth for estimator in classifier.estimators_],
        dtype=int,
    )

    if len(depths) != DEFAULT_N_ESTIMATORS:
        raise RandomForestError(
            "Fitted tree count does not match the fixed protocol."
        )

    return (
        int(len(depths)),
        int(depths.max()),
        float(depths.mean()),
    )


def run_random_forest_experiment(
    dataset: ModelDataset,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> RandomForestResult:
    """Fit on frozen training data and evaluate once on the frozen test data."""
    pipeline = build_random_forest_pipeline(random_state=random_state)

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

    feature_importances, importance_sum = _feature_importance_ranking(
        pipeline,
        retained_features,
    )

    tree_count, maximum_tree_depth, mean_tree_depth = _tree_depth_summary(
        pipeline
    )

    return RandomForestResult(
        policy=dataset.policy,
        label_column=dataset.label_column,
        input_feature_count=len(dataset.feature_columns),
        retained_feature_count=len(retained_features),
        retained_feature_columns=retained_features,
        removed_zero_variance_features=removed_features,
        training_positive_rate=float(dataset.y_train.mean()),
        tree_count=tree_count,
        maximum_tree_depth=maximum_tree_depth,
        mean_tree_depth=mean_tree_depth,
        feature_importance_sum=importance_sum,
        feature_importance_ranking=feature_importances,
        metrics=metrics,
    )


def write_random_forest_result(
    result: RandomForestResult,
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
    """Run fixed Random Forest Davis DTI evaluation from local artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed Random Forest Davis DTI baseline."
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
        help="Fixed random state for tree construction.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/davis_random_forest_cold_drug.json"),
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

        result = run_random_forest_experiment(
            dataset,
            random_state=args.random_state,
        )

        output_path = write_random_forest_result(result, args.output)

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Random Forest evaluation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"Random Forest result written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())