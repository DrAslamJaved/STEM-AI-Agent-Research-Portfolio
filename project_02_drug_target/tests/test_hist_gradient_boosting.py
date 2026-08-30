import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.dataset import ModelDataset
from src.models.hist_gradient_boosting import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_L2_REGULARIZATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FEATURES,
    DEFAULT_MAX_ITER,
    DEFAULT_MAX_LEAF_NODES,
    DEFAULT_MIN_SAMPLES_LEAF,
    HistGradientBoostingResult,
    build_hist_gradient_boosting_pipeline,
    run_hist_gradient_boosting_experiment,
    write_hist_gradient_boosting_result,
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
                "target_id": [
                    f"test_target_{index}" for index in range(6)
                ],
            }
        ),
    )


@pytest.fixture(scope="module")
def boosting_result() -> HistGradientBoostingResult:
    return run_hist_gradient_boosting_experiment(
        synthetic_model_dataset(),
        random_state=7,
    )


def test_pipeline_uses_train_fitted_selector_and_fixed_settings() -> None:
    dataset = synthetic_model_dataset()

    pipeline = build_hist_gradient_boosting_pipeline(random_state=7)

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
    assert classifier.early_stopping is DEFAULT_EARLY_STOPPING
    assert classifier.l2_regularization == DEFAULT_L2_REGULARIZATION
    assert classifier.learning_rate == DEFAULT_LEARNING_RATE
    assert classifier.max_depth == DEFAULT_MAX_DEPTH
    assert classifier.max_features == DEFAULT_MAX_FEATURES
    assert classifier.max_iter == DEFAULT_MAX_ITER
    assert classifier.max_leaf_nodes == DEFAULT_MAX_LEAF_NODES
    assert classifier.min_samples_leaf == DEFAULT_MIN_SAMPLES_LEAF


def test_experiment_reports_fixed_fit_and_selector(
    boosting_result: HistGradientBoostingResult,
) -> None:
    result = boosting_result

    assert result.input_feature_count == 4
    assert result.retained_feature_count == 3
    assert result.removed_zero_variance_features == ("constant",)
    assert result.retained_feature_columns == (
        "signal",
        "weak_signal",
        "noise",
    )

    assert result.fitted_iteration_count == DEFAULT_MAX_ITER
    assert result.trees_per_iteration == 1
    assert result.internal_early_stopping_used is False

    assert 0.0 <= result.metrics.average_precision <= 1.0
    assert 0.0 <= result.metrics.roc_auc <= 1.0


def test_result_is_reproducible_for_fixed_seed(
    boosting_result: HistGradientBoostingResult,
) -> None:
    repeated_result = run_hist_gradient_boosting_experiment(
        synthetic_model_dataset(),
        random_state=7,
    )

    assert (
        boosting_result.metrics.to_dict()
        == repeated_result.metrics.to_dict()
    )

    assert (
        boosting_result.fitted_iteration_count
        == repeated_result.fitted_iteration_count
    )


def test_writer_preserves_model_and_selector_evidence(
    tmp_path: Path,
    boosting_result: HistGradientBoostingResult,
) -> None:
    output_path = write_hist_gradient_boosting_result(
        boosting_result,
        tmp_path / "hist_gradient_boosting.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "HistGradientBoostingClassifier"
    assert payload["retained_feature_count"] == 3
    assert payload["removed_zero_variance_features"] == ["constant"]
    assert payload["fitted_iteration_count"] == DEFAULT_MAX_ITER
    assert payload["internal_early_stopping_used"] is False