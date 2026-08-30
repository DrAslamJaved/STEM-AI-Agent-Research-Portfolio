import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.dataset import ModelDataset
from src.models.logistic_regression import (
    DEFAULT_C,
    DEFAULT_L1_RATIO,
    DEFAULT_MAX_ITER,
    LogisticRegressionError,
    WEIGHTED_PRIMARY_VARIANT,
    build_logistic_pipeline,
    run_logistic_experiment,
    write_logistic_experiment,
)


def synthetic_model_dataset() -> ModelDataset:
    X_train = pd.DataFrame(
        {
            "feature_a": [
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0.2,
                0.5,
                0.9,
                1.1,
                1.5,
                2.0,
                3.0,
            ],
            "feature_b": [0.0, 1.0] * 6,
        }
    )

    y_train = pd.Series([0] * 9 + [1] * 3, dtype="int8")

    X_test = pd.DataFrame(
        {
            "feature_a": [-2.5, -0.8, 0.4, 1.2, 2.5],
            "feature_b": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    y_test = pd.Series([0, 0, 0, 1, 1], dtype="int8")

    return ModelDataset(
        policy="cold_drug",
        label_column="interaction_kd_le_1000_nM",
        feature_columns=tuple(X_train.columns),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_metadata=pd.DataFrame(
            {
                "observed_pair_index": range(len(X_train)),
                "drug_id": ["train_drug"] * len(X_train),
                "target_id": ["target"] * len(X_train),
            }
        ),
        test_metadata=pd.DataFrame(
            {
                "observed_pair_index": range(
                    100,
                    100 + len(X_test),
                ),
                "drug_id": ["test_drug"] * len(X_test),
                "target_id": ["target"] * len(X_test),
            }
        ),
    )


def test_pipeline_uses_train_only_scaling_and_fixed_parameters() -> None:
    dataset = synthetic_model_dataset()

    pipeline = build_logistic_pipeline(
        "balanced",
        random_state=7,
    )

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]

    pipeline.fit(dataset.X_train, dataset.y_train)

    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    assert np.allclose(
        scaler.mean_,
        dataset.X_train.mean().to_numpy(),
    )

    assert classifier.class_weight == "balanced"
    assert classifier.C == DEFAULT_C
    assert classifier.l1_ratio == DEFAULT_L1_RATIO
    assert classifier.max_iter == DEFAULT_MAX_ITER
    assert classifier.solver == "liblinear"


def test_experiment_reports_weighted_primary_and_unweighted_sensitivity() -> None:
    result = run_logistic_experiment(
        synthetic_model_dataset(),
        random_state=7,
    )

    assert result.primary_variant == WEIGHTED_PRIMARY_VARIANT
    assert len(result.results) == 2

    weighted, unweighted = result.results

    assert weighted.class_weight == "balanced"
    assert unweighted.class_weight is None

    assert weighted.feature_count == 2

    assert set(weighted.standardized_coefficients) == {
        "feature_a",
        "feature_b",
    }

    assert 0.0 <= weighted.metrics.average_precision <= 1.0
    assert 0.0 <= unweighted.metrics.average_precision <= 1.0


def test_invalid_class_weight_fails_clearly() -> None:
    with pytest.raises(LogisticRegressionError, match="balanced"):
        build_logistic_pipeline("unsupported")


def test_writer_preserves_both_pre_specified_variants(
    tmp_path: Path,
) -> None:
    result = run_logistic_experiment(
        synthetic_model_dataset(),
        random_state=7,
    )

    output_path = write_logistic_experiment(
        result,
        tmp_path / "logistic.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["primary_variant"] == WEIGHTED_PRIMARY_VARIANT
    assert len(payload["results"]) == 2

    assert (
        payload["results"][0]["model_parameters"]["class_weight"]
        == "balanced"
    )

    assert (
        payload["results"][1]["model_parameters"]["class_weight"]
        is None
    )