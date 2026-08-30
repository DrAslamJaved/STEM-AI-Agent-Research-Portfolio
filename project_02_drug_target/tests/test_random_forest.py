import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.dataset import ModelDataset
from src.models.random_forest import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_N_JOBS,
    RandomForestResult,
    build_random_forest_pipeline,
    run_random_forest_experiment,
    write_random_forest_result,
)


def synthetic_model_dataset() -> ModelDataset:
    feature_columns = (
        "constant",
        "signal",
        "weak_signal",
        "noise",
    )

    X_train = pd.DataFrame(
        {
            "constant": [1.0] * 24,
            "signal": [0.0] * 12 + [1.0] * 12,
            "weak_signal": [0.0, 1.0] * 12,
            "noise": [0.0, 1.0, 2.0, 3.0] * 6,
        }
    )

    # The training selector must remove "constant" even though it varies here.
    X_test = pd.DataFrame(
        {
            "constant": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "signal": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "weak_signal": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "noise": [3.0, 2.0, 1.0, 0.0, 3.0, 2.0],
        }
    )

    return ModelDataset(
        policy="cold_drug",
        label_column="interaction_kd_le_1000_nM",
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=pd.Series([0] * 12 + [1] * 12, dtype="int8"),
        X_test=X_test,
        y_test=pd.Series([0, 0, 0, 1, 1, 1], dtype="int8"),
        train_metadata=pd.DataFrame(
            {
                "observed_pair_index": range(24),
                "drug_id": ["train_drug"] * 24,
                "target_id": [f"target_{index}" for index in range(24)],
            }
        ),
        test_metadata=pd.DataFrame(
            {
                "observed_pair_index": range(24, 30),
                "drug_id": ["test_drug"] * 6,
                "target_id": [f"test_target_{index}" for index in range(6)],
            }
        ),
    )


@pytest.fixture(scope="module")
def random_forest_result() -> RandomForestResult:
    return run_random_forest_experiment(
        synthetic_model_dataset(),
        random_state=7,
    )


def test_pipeline_uses_train_fitted_variance_selector_and_fixed_settings() -> None:
    dataset = synthetic_model_dataset()

    pipeline = build_random_forest_pipeline(random_state=7)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [
        "variance_threshold",
        "classifier",
    ]

    pipeline.fit(dataset.X_train, dataset.y_train)

    selector = pipeline.named_steps["variance_threshold"]
    classifier = pipeline.named_steps["classifier"]

    assert selector.threshold == 0.0
    assert selector.get_support().tolist() == [False, True, True, True]

    assert classifier.class_weight == DEFAULT_CLASS_WEIGHT
    assert classifier.max_depth == DEFAULT_MAX_DEPTH
    assert classifier.max_features == DEFAULT_MAX_FEATURES
    assert classifier.min_samples_leaf == DEFAULT_MIN_SAMPLES_LEAF
    assert classifier.n_estimators == DEFAULT_N_ESTIMATORS
    assert classifier.n_jobs == DEFAULT_N_JOBS


def test_experiment_reports_selector_and_aligned_importances(
    random_forest_result: RandomForestResult,
) -> None:
    result = random_forest_result

    assert result.input_feature_count == 4
    assert result.retained_feature_count == 3
    assert result.removed_zero_variance_features == ("constant",)
    assert result.retained_feature_columns == (
        "signal",
        "weak_signal",
        "noise",
    )

    assert result.tree_count == DEFAULT_N_ESTIMATORS
    assert result.maximum_tree_depth <= DEFAULT_MAX_DEPTH
    assert result.feature_importance_sum == pytest.approx(1.0)

    ranked_features = {
        entry.feature for entry in result.feature_importance_ranking
    }

    assert ranked_features == {"signal", "weak_signal", "noise"}
    assert 0.0 <= result.metrics.average_precision <= 1.0
    assert 0.0 <= result.metrics.roc_auc <= 1.0


def test_result_is_reproducible_for_the_fixed_seed(
    random_forest_result: RandomForestResult,
) -> None:
    repeated_result = run_random_forest_experiment(
        synthetic_model_dataset(),
        random_state=7,
    )

    assert (
        random_forest_result.metrics.to_dict()
        == repeated_result.metrics.to_dict()
    )

    assert (
        random_forest_result.feature_importance_ranking
        == repeated_result.feature_importance_ranking
    )


def test_writer_preserves_model_and_selector_evidence(
    tmp_path: Path,
    random_forest_result: RandomForestResult,
) -> None:
    output_path = write_random_forest_result(
        random_forest_result,
        tmp_path / "random_forest.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "RandomForestClassifier"
    assert payload["retained_feature_count"] == 3
    assert payload["removed_zero_variance_features"] == ["constant"]
    assert payload["tree_count"] == DEFAULT_N_ESTIMATORS
    assert payload["feature_importance_sum"] == pytest.approx(1.0)