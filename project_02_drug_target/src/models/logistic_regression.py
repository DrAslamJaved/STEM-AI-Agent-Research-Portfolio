"""Run fixed, interpretable logistic-regression DTI baselines."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.dataset import ModelDataset, load_train_test_data
from src.models.evaluation import (
    BinaryClassificationMetrics,
    DEFAULT_DECISION_THRESHOLD,
    evaluate_binary_classification,
)


DEFAULT_RANDOM_STATE = 20260830
DEFAULT_C = 1.0
DEFAULT_L1_RATIO = 0.0
DEFAULT_MAX_ITER = 1000
DEFAULT_SOLVER = "liblinear"

WEIGHTED_PRIMARY_VARIANT = "class_weight_balanced_primary"
UNWEIGHTED_SENSITIVITY_VARIANT = "class_weight_none_sensitivity"


class LogisticRegressionError(ValueError):
    """Raised when the fixed logistic-regression protocol cannot be followed."""


@dataclass(frozen=True)
class LogisticVariantResult:
    """Serializable evaluation and coefficient record for one fixed variant."""

    variant: str
    class_weight: str | None
    random_state: int
    policy: str
    label_column: str
    feature_count: int
    training_positive_rate: float
    intercept: float
    standardized_coefficients: dict[str, float]
    iteration_count: int
    metrics: BinaryClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible variant report."""
        return {
            "variant": self.variant,
            "model_name": "LogisticRegression",
            "model_parameters": {
                "C": DEFAULT_C,
                "class_weight": self.class_weight,
                "l1_ratio": DEFAULT_L1_RATIO,
                "max_iter": DEFAULT_MAX_ITER,
                "random_state": self.random_state,
                "regularization": "L2",
                "solver": DEFAULT_SOLVER,
                "standardization": "StandardScaler",
            },
            "policy": self.policy,
            "label_column": self.label_column,
            "feature_count": self.feature_count,
            "training_positive_rate": self.training_positive_rate,
            "intercept": self.intercept,
            "standardized_coefficients": self.standardized_coefficients,
            "iteration_count": self.iteration_count,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class LogisticExperimentResult:
    """Pre-specified primary and sensitivity logistic-regression results."""

    primary_variant: str
    sensitivity_variant: str
    results: tuple[LogisticVariantResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible experiment report."""
        return {
            "primary_variant": self.primary_variant,
            "sensitivity_variant": self.sensitivity_variant,
            "results": [result.to_dict() for result in self.results],
        }


def build_logistic_pipeline(
    class_weight: str | None,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build an unfitted train-only scaler plus fixed logistic classifier."""
    if class_weight not in ("balanced", None):
        raise LogisticRegressionError(
            "class_weight must be either 'balanced' or None."
        )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=DEFAULT_C,
                    class_weight=class_weight,
                    l1_ratio=DEFAULT_L1_RATIO,
                    max_iter=DEFAULT_MAX_ITER,
                    random_state=int(random_state),
                    solver=DEFAULT_SOLVER,
                ),
            ),
        ]
    )


def _positive_class_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return class-1 probabilities without assuming a fixed class order."""
    classifier = pipeline.named_steps["classifier"]

    positive_positions = np.flatnonzero(classifier.classes_ == 1)

    if len(positive_positions) != 1:
        raise LogisticRegressionError(
            "Fitted classifier must contain exactly one positive class."
        )

    probabilities = pipeline.predict_proba(features)

    positive_scores = probabilities[:, int(positive_positions[0])]

    if len(positive_scores) != len(features):
        raise LogisticRegressionError(
            "Classifier returned an unexpected number of probabilities."
        )

    if not np.isfinite(positive_scores).all():
        raise LogisticRegressionError(
            "Classifier returned non-finite positive probabilities."
        )

    return positive_scores.astype(float)


def _standardized_coefficients(
    pipeline: Pipeline,
    feature_columns: tuple[str, ...],
) -> tuple[float, dict[str, float], int]:
    """Extract binary-model coefficients in the standardized feature space."""
    classifier = pipeline.named_steps["classifier"]

    if classifier.coef_.shape != (1, len(feature_columns)):
        raise LogisticRegressionError(
            "Classifier coefficient shape does not match the feature columns."
        )

    if classifier.intercept_.shape != (1,):
        raise LogisticRegressionError(
            "Classifier intercept shape is invalid for binary classification."
        )

    if classifier.n_iter_.shape != (1,):
        raise LogisticRegressionError(
            "Classifier iteration-count shape is invalid for binary classification."
        )

    coefficients = {
        column: float(value)
        for column, value in zip(
            feature_columns,
            classifier.coef_[0],
            strict=True,
        )
    }

    return (
        float(classifier.intercept_[0]),
        coefficients,
        int(classifier.n_iter_[0]),
    )


def run_logistic_variant(
    dataset: ModelDataset,
    *,
    variant: str,
    class_weight: str | None,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> LogisticVariantResult:
    """Fit and evaluate one fixed logistic-regression variant."""
    pipeline = build_logistic_pipeline(
        class_weight,
        random_state=random_state,
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=ConvergenceWarning,
            )
            pipeline.fit(dataset.X_train, dataset.y_train)

    except ConvergenceWarning as error:
        raise LogisticRegressionError(
            "Logistic regression did not converge within the fixed max_iter."
        ) from error

    positive_scores = _positive_class_probabilities(
        pipeline,
        dataset.X_test,
    )

    metrics = evaluate_binary_classification(
        dataset.y_test,
        positive_scores,
        decision_threshold=decision_threshold,
    )

    intercept, coefficients, iteration_count = _standardized_coefficients(
        pipeline,
        dataset.feature_columns,
    )

    return LogisticVariantResult(
        variant=variant,
        class_weight=class_weight,
        random_state=int(random_state),
        policy=dataset.policy,
        label_column=dataset.label_column,
        feature_count=len(dataset.feature_columns),
        training_positive_rate=float(dataset.y_train.mean()),
        intercept=intercept,
        standardized_coefficients=coefficients,
        iteration_count=iteration_count,
        metrics=metrics,
    )


def run_logistic_experiment(
    dataset: ModelDataset,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> LogisticExperimentResult:
    """Run the pre-specified weighted and unweighted comparisons."""
    weighted_primary = run_logistic_variant(
        dataset,
        variant=WEIGHTED_PRIMARY_VARIANT,
        class_weight="balanced",
        random_state=random_state,
        decision_threshold=decision_threshold,
    )

    unweighted_sensitivity = run_logistic_variant(
        dataset,
        variant=UNWEIGHTED_SENSITIVITY_VARIANT,
        class_weight=None,
        random_state=random_state,
        decision_threshold=decision_threshold,
    )

    return LogisticExperimentResult(
        primary_variant=WEIGHTED_PRIMARY_VARIANT,
        sensitivity_variant=UNWEIGHTED_SENSITIVITY_VARIANT,
        results=(weighted_primary, unweighted_sensitivity),
    )


def write_logistic_experiment(
    result: LogisticExperimentResult,
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
    """Run fixed logistic-regression baselines from local Davis artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate fixed logistic-regression Davis DTI baselines."
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
        default="cold_drug",
        choices=("random_pair", "cold_drug", "cold_target"),
        help="Frozen split policy to evaluate.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Fixed random state for the liblinear solver.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/davis_logistic_regression_cold_drug.json"),
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

        result = run_logistic_experiment(
            dataset,
            random_state=args.random_state,
        )

        output_path = write_logistic_experiment(result, args.output)

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(
            f"Logistic-regression evaluation failed: {error}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"Logistic-regression result written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())