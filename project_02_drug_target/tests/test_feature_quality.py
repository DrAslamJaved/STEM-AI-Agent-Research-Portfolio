import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.quality import (
    FeatureQualityError,
    audit_training_feature_quality,
    write_feature_quality_audit,
)
from src.models.dataset import ModelDataset


def synthetic_model_dataset() -> ModelDataset:
    feature_columns = ("constant", "linear", "twice_linear")

    X_train = pd.DataFrame(
        {
            "constant": [2.0, 2.0, 2.0, 2.0],
            "linear": [0.0, 1.0, 2.0, 3.0],
            "twice_linear": [0.0, 2.0, 4.0, 6.0],
        }
    )

    # Deliberately invalid test values prove that the audit reads X_train only.
    X_test = pd.DataFrame(
        {
            "constant": [np.nan, np.inf],
            "linear": [np.inf, -np.inf],
            "twice_linear": [np.nan, np.inf],
        }
    )

    return ModelDataset(
        policy="cold_drug",
        label_column="interaction_kd_le_1000_nM",
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=pd.Series([0, 0, 1, 1], dtype="int8"),
        X_test=X_test,
        y_test=pd.Series([0, 1], dtype="int8"),
        train_metadata=pd.DataFrame(
            {
                "observed_pair_index": [0, 1, 2, 3],
                "drug_id": ["train_drug"] * 4,
                "target_id": ["target_a", "target_b"] * 2,
            }
        ),
        test_metadata=pd.DataFrame(
            {
                "observed_pair_index": [4, 5],
                "drug_id": ["test_drug"] * 2,
                "target_id": ["target_a", "target_b"],
            }
        ),
    )


def test_audit_uses_training_features_only() -> None:
    audit = audit_training_feature_quality(synthetic_model_dataset())

    assert audit.audit_partition == "train"
    assert audit.audit_pair_count == 4
    assert audit.input_feature_count == 3
    assert audit.usable_feature_count == 2

    assert audit.zero_variance_features == ("constant",)
    assert audit.retained_feature_columns == (
        "linear",
        "twice_linear",
    )

    assert audit.missing_feature_value_count == 0
    assert audit.nonfinite_feature_value_count == 0

    assert audit.high_correlation_pair_count == 1

    pair = audit.high_correlation_pairs[0]

    assert pair.feature_a == "linear"
    assert pair.feature_b == "twice_linear"
    assert pair.pearson_correlation == pytest.approx(1.0)
    assert pair.absolute_pearson_correlation == pytest.approx(1.0)


def test_training_missing_values_fail_clearly() -> None:
    dataset = synthetic_model_dataset()
    broken_training_features = dataset.X_train.copy()
    broken_training_features.loc[0, "linear"] = np.nan

    broken_dataset = replace(
        dataset,
        X_train=broken_training_features,
    )

    with pytest.raises(FeatureQualityError, match="missing"):
        audit_training_feature_quality(broken_dataset)


def test_training_nonfinite_values_fail_clearly() -> None:
    dataset = synthetic_model_dataset()
    broken_training_features = dataset.X_train.copy()
    broken_training_features.loc[0, "linear"] = np.inf

    broken_dataset = replace(
        dataset,
        X_train=broken_training_features,
    )

    with pytest.raises(FeatureQualityError, match="non-finite"):
        audit_training_feature_quality(broken_dataset)


def test_invalid_correlation_threshold_fails_clearly() -> None:
    with pytest.raises(FeatureQualityError, match="interval"):
        audit_training_feature_quality(
            synthetic_model_dataset(),
            high_correlation_threshold=0.0,
        )


def test_feature_contract_mismatch_fails_clearly() -> None:
    dataset = synthetic_model_dataset()
    mismatched_training_features = dataset.X_train.rename(
        columns={"linear": "renamed_linear"}
    )

    broken_dataset = replace(
        dataset,
        X_train=mismatched_training_features,
    )

    with pytest.raises(FeatureQualityError, match="frozen feature contract"):
        audit_training_feature_quality(broken_dataset)


def test_writer_preserves_train_only_audit(tmp_path: Path) -> None:
    audit = audit_training_feature_quality(synthetic_model_dataset())

    output_path = write_feature_quality_audit(
        audit,
        tmp_path / "feature_quality.json",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["audit_partition"] == "train"
    assert payload["zero_variance_features"] == ["constant"]
    assert payload["high_correlation_pair_count"] == 1
    assert payload["high_correlation_pairs"][0]["feature_a"] == "linear"