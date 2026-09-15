"""Reproducible DTI baselines using transparent fixed features."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dti_features import feature_matrix
from .evaluation import evaluate_binary_predictions


def _labels(records: Iterable[Mapping[str, Any]]) -> list[int]:
    values = [int(row["label"]) for row in records]
    if not values or any(value not in (0, 1) for value in values):
        raise ValueError("Non-empty binary DTI labels are required.")
    return values


def _metric_record(labels: list[int], scores: list[float]) -> dict[str, Any]:
    result = evaluate_binary_predictions(labels, scores).to_dict()
    result["brier_score"] = brier_score_loss(labels, scores)
    return result


def evaluate_dti_models(train_records: Iterable[Mapping[str, Any]], test_records: Iterable[Mapping[str, Any]], *, seed: int = 20260915) -> dict[str, dict[str, Any]]:
    """Evaluate prevalence, logistic, and random-forest baselines on one split."""
    train, test = list(train_records), list(test_records)
    y_train, y_test = _labels(train), _labels(test)
    if len(set(y_train)) < 2:
        raise ValueError("Training partition must contain both binary classes.")
    x_train, x_test = feature_matrix(train), feature_matrix(test)
    prevalence = sum(y_train) / len(y_train)
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])
    forest = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1,
    )
    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    return {
        "prevalence": _metric_record(y_test, [prevalence] * len(y_test)),
        "logistic_regression": _metric_record(y_test, logistic.predict_proba(x_test)[:, 1].tolist()),
        "random_forest": _metric_record(y_test, forest.predict_proba(x_test)[:, 1].tolist()),
    }
