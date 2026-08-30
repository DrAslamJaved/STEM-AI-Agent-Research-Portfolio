import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.models.baselines import (
    DEFAULT_RANDOM_STATE,
    run_dummy_prior_baseline,
    write_baseline_result,
)
from src.models.dataset import ModelDataset


def synthetic_model_dataset() -> ModelDataset:
    return ModelDataset(
        policy="cold_drug",
        label_column="interaction_kd_le_1000_nM",
        feature_columns=("feature_a", "feature_b"),
        X_train=pd.DataFrame(
            {
                "feature_a": [0.0, 1.0, 2.0, 3.0],
                "feature_b": [3.0, 2.0, 1.0, 0.0],
            }
        ),
        y_train=pd.Series([0, 0, 0, 1], dtype="int8"),
        X_test=pd.DataFrame(
            {
                "feature_a": [5.0, -1.0, 7.0, 4.0],
                "feature_b": [2.0, 8.0, 0.0, 3.0],
            }
        ),
        y_test=pd.Series([0, 0, 1, 1], dtype="int8"),
        train_metadata=pd.DataFrame(
            {
                "observed_pair_index": [0, 1, 2, 3],
                "drug_id": ["drug_a", "drug_a", "drug_b", "drug_b"],
                "target_id": [
                    "target_a",
                    "target_b",
                    "target_a",
                    "target_b",
                ],
            }
        ),
        test_metadata=pd.DataFrame(
            {
                "observed_pair_index": [4, 5, 6, 7],
                "drug_id": ["drug_c", "drug_c", "drug_d", "drug_d"],
                "target_id": [
                    "target_a",
                    "target_b",
                    "target_a",
                    "target_b",
                ],
            }
        ),
    )


def test_prior_baseline_uses_training_class_prevalence_only() -> None:
    result = run_dummy_prior_baseline(synthetic_model_dataset())

    assert result.model_name == "DummyClassifier"
    assert result.strategy == "prior"
    assert result.random_state == DEFAULT_RANDOM_STATE

    assert result.training_positive_rate == pytest.approx(0.25)
    assert result.test_positive_probability_min == pytest.approx(0.25)
    assert result.test_positive_probability_max == pytest.approx(0.25)

    assert result.metrics.roc_auc == pytest.approx(0.5)
    assert result.metrics.average_precision == pytest.approx(0.5)
    assert result.metrics.accuracy == pytest.approx(0.5)
    assert result.metrics.precision == 0.0
    assert result.metrics.recall == 0.0
    assert result.metrics.f1 == 0.0

    assert (
        result.metrics.true_negative,
        result.metrics.false_positive,
        result.metrics.false_negative,
        result.metrics.true_positive,
    ) == (2, 0, 2, 0)


def test_prior_baseline_ignores_test_feature_values() -> None:
    dataset = synthetic_model_dataset()

    changed_features = pd.DataFrame(
        {
            "feature_a": [1000.0, -500.0, 12.0, 9.0],
            "feature_b": [-10.0, 400.0, 5.0, 8.0],
        }
    )

    original = run_dummy_prior_baseline(dataset)

    changed = run_dummy_prior_baseline(
        replace(dataset, X_test=changed_features)
    )

    assert changed.to_dict() == original.to_dict()


def test_baseline_writer_preserves_reproducibility_evidence(
    tmp_path: Path,
) -> None:
    result = run_dummy_prior_baseline(
        synthetic_model_dataset(),
        random_state=7,
    )

    output_path = write_baseline_result(
        result,
        tmp_path / "baseline.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "DummyClassifier"

    assert payload["model_parameters"] == {
        "random_state": 7,
        "strategy": "prior",
    }

    assert payload["training_positive_rate"] == pytest.approx(0.25)

    assert payload["metrics"]["average_precision"] == pytest.approx(0.5)