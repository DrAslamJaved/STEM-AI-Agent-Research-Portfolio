"""Run simple, reproducible binary DTI baseline models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.models.dataset import ModelDataset, load_train_test_data
from src.models.evaluation import (
    BinaryClassificationMetrics,
    DEFAULT_DECISION_THRESHOLD,
    evaluate_binary_classification,
)


DUMMY_STRATEGY = "prior"
DEFAULT_RANDOM_STATE = 20260830


class BaselineError(ValueError):
    """Raised when a simple baseline cannot be fit or evaluated safely."""


@dataclass(frozen=True)
class BaselineResult:
    """Serializable result of one deterministic prior-probability baseline."""

    model_name: str
    strategy: str
    random_state: int
    policy: str
    label_column: str
    training_positive_rate: float
    test_positive_probability_min: float
    test_positive_probability_max: float
    metrics: BinaryClassificationMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible model report."""
        return {
            "model_name": self.model_name,
            "model_parameters": {
                "random_state": self.random_state,
                "strategy": self.strategy,
            },
            "policy": self.policy,
            "label_column": self.label_column,
            "training_positive_rate": self.training_positive_rate,
            "test_positive_probability_min": (
                self.test_positive_probability_min
            ),
            "test_positive_probability_max": (
                self.test_positive_probability_max
            ),
            "metrics": self.metrics.to_dict(),
        }


def _positive_class_probabilities(
    model: DummyClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return positive-class probabilities without assuming class order."""
    if not hasattr(model, "classes_"):
        raise BaselineError(
            "DummyClassifier must be fitted before prediction."
        )

    positive_positions = np.flatnonzero(model.classes_ == 1)

    if len(positive_positions) != 1:
        raise BaselineError(
            "Fitted DummyClassifier must contain exactly one positive class."
        )

    probabilities = model.predict_proba(features)

    positive_scores = probabilities[:, int(positive_positions[0])]

    if len(positive_scores) != len(features):
        raise BaselineError(
            "DummyClassifier returned an unexpected number of probabilities."
        )

    if not np.isfinite(positive_scores).all():
        raise BaselineError(
            "DummyClassifier returned non-finite positive probabilities."
        )

    return positive_scores.astype(float)


def run_dummy_prior_baseline(
    dataset: ModelDataset,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> BaselineResult:
    """Fit the empirical training-prior baseline and evaluate frozen test data.

    The model deliberately ignores chemical and protein feature values. It
    learns only the prevalence of positive labels in the training partition.
    """
    state = int(random_state)

    model = DummyClassifier(
        strategy=DUMMY_STRATEGY,
        random_state=state,
    )

    model.fit(dataset.X_train, dataset.y_train)

    positive_scores = _positive_class_probabilities(model, dataset.X_test)

    metrics = evaluate_binary_classification(
        dataset.y_test,
        positive_scores,
        decision_threshold=decision_threshold,
    )

    return BaselineResult(
        model_name="DummyClassifier",
        strategy=DUMMY_STRATEGY,
        random_state=state,
        policy=dataset.policy,
        label_column=dataset.label_column,
        training_positive_rate=float(dataset.y_train.mean()),
        test_positive_probability_min=float(positive_scores.min()),
        test_positive_probability_max=float(positive_scores.max()),
        metrics=metrics,
    )


def write_baseline_result(
    result: BaselineResult,
    output_path: str | Path,
) -> Path:
    """Write compact baseline evidence without serializing a model object."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return destination


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic DummyClassifier baseline from local artifacts."""
    parser = argparse.ArgumentParser(
        description="Evaluate a prior-probability Davis DTI baseline."
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
        help="Recorded seed; the prior strategy itself is deterministic.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/davis_dummy_baseline_cold_drug.json"),
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

        result = run_dummy_prior_baseline(
            dataset,
            random_state=args.random_state,
        )

        output_path = write_baseline_result(result, args.output)

    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        print(f"Baseline evaluation failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"Baseline result written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())