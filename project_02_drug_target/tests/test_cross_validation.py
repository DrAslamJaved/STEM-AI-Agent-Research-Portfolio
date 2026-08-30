import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.cross_validation import (
    CANDIDATE_MODEL_IDS,
    CrossValidationError,
    InnerCVRun,
    run_inner_cold_drug_cv,
    write_inner_cv_summary,
    write_oof_predictions,
)
from src.models.dataset import ModelDataset


def synthetic_model_dataset() -> ModelDataset:
    feature_columns = (
        "constant",
        "signal",
        "weak_signal",
        "noise",
    )

    training_rows: list[dict[str, float]] = []
    training_labels: list[int] = []
    training_metadata: list[dict[str, object]] = []

    observed_pair_index = 0

    for drug_index in range(10):
        for target_index in range(4):
            label = int(target_index >= 2)

            training_rows.append(
                {
                    "constant": 1.0,
                    "signal": float(label),
                    "weak_signal": float(
                        (drug_index + target_index) % 2
                    ),
                    "noise": float(target_index),
                }
            )

            training_labels.append(label)

            training_metadata.append(
                {
                    "observed_pair_index": observed_pair_index,
                    "drug_id": f"train_drug_{drug_index}",
                    "target_id": f"target_{target_index}",
                }
            )

            observed_pair_index += 1

    X_train = pd.DataFrame(training_rows)

    # Deliberately invalid outer-test values prove CV does not access X_test.
    X_test = pd.DataFrame(
        {
            "constant": [np.nan, np.inf],
            "signal": [np.inf, np.nan],
            "weak_signal": [np.nan, np.inf],
            "noise": [np.inf, np.nan],
        }
    )

    return ModelDataset(
        policy="cold_drug",
        label_column="interaction_kd_le_1000_nM",
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=pd.Series(training_labels, dtype="int8"),
        X_test=X_test,
        y_test=pd.Series([0, 1], dtype="int8"),
        train_metadata=pd.DataFrame(training_metadata),
        test_metadata=pd.DataFrame(
            {
                "observed_pair_index": [1000, 1001],
                "drug_id": ["outer_test_drug", "outer_test_drug"],
                "target_id": ["target_0", "target_1"],
            }
        ),
    )


@pytest.fixture(scope="module")
def inner_cv_run() -> InnerCVRun:
    return run_inner_cold_drug_cv(
        synthetic_model_dataset(),
        n_splits=2,
        random_state=7,
    )


def test_inner_cv_has_drug_disjoint_validation_folds(
    inner_cv_run: InnerCVRun,
) -> None:
    summary = inner_cv_run.summary

    assert summary.cv_scope == "frozen_outer_training_partition_only"
    assert summary.outer_test_partition_used is False
    assert summary.input_pair_count == 40
    assert summary.input_drug_count == 10
    assert summary.n_splits == 2

    assert tuple(
        result.model_id for result in summary.model_results
    ) == CANDIDATE_MODEL_IDS

    for candidate_result in summary.model_results:
        assert len(candidate_result.fold_results) == 2
        assert candidate_result.oof_prediction_count == 40

        for fold_result in candidate_result.fold_results:
            assert fold_result.drug_overlap_count == 0
            assert fold_result.train_drug_count == 5
            assert fold_result.validation_drug_count == 5
            assert fold_result.train_pair_count == 20
            assert fold_result.validation_pair_count == 20

    feature_counts = {
        candidate_result.model_id: {
            fold_result.feature_count_after_preprocessing
            for fold_result in candidate_result.fold_results
        }
        for candidate_result in summary.model_results
    }

    assert feature_counts["dummy_prior"] == {4}
    assert feature_counts["logistic_regression_balanced"] == {4}
    assert feature_counts["random_forest_balanced"] == {3}
    assert feature_counts["hist_gradient_boosting_balanced"] == {3}


def test_oof_predictions_cover_each_training_pair_once_per_model(
    inner_cv_run: InnerCVRun,
) -> None:
    oof_predictions = inner_cv_run.oof_predictions

    assert tuple(oof_predictions.columns) == (
        "model_id",
        "model_name",
        "fold_index",
        "fit_random_state",
        "observed_pair_index",
        "drug_id",
        "target_id",
        "y_true",
        "positive_probability",
    )

    assert len(oof_predictions) == 160

    for model_id in CANDIDATE_MODEL_IDS:
        candidate_predictions = oof_predictions.loc[
            oof_predictions["model_id"].eq(model_id)
        ]

        assert len(candidate_predictions) == 40
        assert candidate_predictions["observed_pair_index"].is_unique
        assert set(candidate_predictions["observed_pair_index"]) == set(
            range(40)
        )
        assert set(candidate_predictions["fold_index"]) == {0, 1}
        assert np.isfinite(
            candidate_predictions["positive_probability"].to_numpy()
        ).all()


def test_inner_cv_is_reproducible_for_fixed_seed(
    inner_cv_run: InnerCVRun,
) -> None:
    repeated_run = run_inner_cold_drug_cv(
        synthetic_model_dataset(),
        n_splits=2,
        random_state=7,
    )

    assert inner_cv_run.summary.to_dict() == repeated_run.summary.to_dict()

    pd.testing.assert_frame_equal(
        inner_cv_run.oof_predictions,
        repeated_run.oof_predictions,
    )


def test_writers_preserve_summary_and_oof_contract(
    tmp_path: Path,
    inner_cv_run: InnerCVRun,
) -> None:
    summary_path = write_inner_cv_summary(
        inner_cv_run.summary,
        tmp_path / "inner_cv.json",
    )

    oof_path = write_oof_predictions(
        inner_cv_run.oof_predictions,
        tmp_path / "oof_predictions.csv",
    )

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    restored_oof = pd.read_csv(oof_path)

    assert summary_payload["outer_test_partition_used"] is False
    assert summary_payload["n_splits"] == 2
    assert len(summary_payload["model_results"]) == 4
    assert tuple(restored_oof.columns) == tuple(
        inner_cv_run.oof_predictions.columns
    )
    assert len(restored_oof) == 160


def test_too_many_folds_fail_before_model_fitting() -> None:
    with pytest.raises(
        CrossValidationError,
        match="cannot exceed",
    ):
        run_inner_cold_drug_cv(
            synthetic_model_dataset(),
            n_splits=11,
            random_state=7,
        )