"""Evaluate binary DTI predictions using pre-specified metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_DECISION_THRESHOLD = 0.50


class EvaluationError(ValueError):
    """Raised when binary evaluation inputs are invalid or ambiguous."""


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Threshold-free and threshold-dependent binary evaluation results."""

    decision_threshold: float
    sample_count: int
    positive_count: int
    negative_count: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metric record."""
        return asdict(self)


def _as_one_dimensional_array(
    values: Sequence[object] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Return a non-empty one-dimensional NumPy array."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise EvaluationError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise EvaluationError(f"{name} must not be empty.")

    return array


def _validated_binary_labels(
    values: Sequence[object] | np.ndarray,
) -> np.ndarray:
    """Return finite binary labels encoded as int8."""
    labels = _as_one_dimensional_array(values, "y_true")

    try:
        numeric_labels = labels.astype(float)
    except (TypeError, ValueError) as error:
        raise EvaluationError(
            "y_true must contain numeric binary labels."
        ) from error

    if not np.isfinite(numeric_labels).all():
        raise EvaluationError(
            "y_true must contain finite binary labels."
        )

    if not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise EvaluationError(
            "y_true must contain only 0 and 1 labels."
        )

    if np.unique(numeric_labels).size != 2:
        raise EvaluationError(
            "y_true must contain both positive and negative examples."
        )

    return numeric_labels.astype(np.int8)


def _validated_probability_scores(
    values: Sequence[object] | np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Return finite positive-class probabilities in the unit interval."""
    scores = _as_one_dimensional_array(values, "y_score")

    if len(scores) != expected_length:
        raise EvaluationError(
            "y_true and y_score must have the same length."
        )

    try:
        numeric_scores = scores.astype(float)
    except (TypeError, ValueError) as error:
        raise EvaluationError(
            "y_score must contain numeric probabilities."
        ) from error

    if not np.isfinite(numeric_scores).all():
        raise EvaluationError(
            "y_score must contain finite probabilities."
        )

    if (numeric_scores < 0.0).any() or (numeric_scores > 1.0).any():
        raise EvaluationError(
            "y_score probabilities must lie between 0 and 1."
        )

    return numeric_scores


def _validated_threshold(threshold: float) -> float:
    """Return a finite probability threshold in the unit interval."""
    try:
        value = float(threshold)
    except (TypeError, ValueError) as error:
        raise EvaluationError(
            "decision_threshold must be numeric."
        ) from error

    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EvaluationError(
            "decision_threshold must lie between 0 and 1."
        )

    return value


def evaluate_binary_classification(
    y_true: Sequence[object] | np.ndarray,
    y_score: Sequence[object] | np.ndarray,
    *,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> BinaryClassificationMetrics:
    """Evaluate positive-class probabilities against binary labels.

    Average precision is the pre-specified primary ranking metric. Accuracy,
    precision, recall, F1, and the confusion matrix use a fixed threshold and
    must not be optimized on the outer test set.
    """
    labels = _validated_binary_labels(y_true)
    scores = _validated_probability_scores(y_score, len(labels))
    threshold = _validated_threshold(decision_threshold)

    predictions = (scores >= threshold).astype(np.int8)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    )

    positive_count = int(labels.sum())
    sample_count = int(len(labels))

    return BinaryClassificationMetrics(
        decision_threshold=threshold,
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=int(sample_count - positive_count),
        positive_rate=float(positive_count / sample_count),
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(
            precision_score(labels, predictions, zero_division=0)
        ),
        recall=float(
            recall_score(labels, predictions, zero_division=0)
        ),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
    )