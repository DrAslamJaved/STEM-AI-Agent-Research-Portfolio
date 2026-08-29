"""Tests for leakage-safe UNSW-NB15 preprocessing."""

from dataclasses import fields, is_dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from cyber_pca.unsw_data import (
    UNSWNB15Data,
    UNSW_CURATED_COLUMNS,
)
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWRawDataSplits,
    UNSWStandardizedDataSplits,
    build_unsw_preprocessing_evidence,
    split_unsw_normal_calibration_test,
    standardize_unsw_splits,
    write_unsw_preprocessing_evidence,
)


def test_unsw_preprocessing_public_interface() -> None:
    assert is_dataclass(UNSWRawDataSplits)
    assert is_dataclass(UNSWPreprocessor)
    assert is_dataclass(
        UNSWStandardizedDataSplits
    )

    assert [
        field.name
        for field in fields(UNSWRawDataSplits)
    ] == [
        "normal_fit",
        "normal_calibration",
        "test",
    ]

    assert [
        field.name
        for field in fields(UNSWPreprocessor)
    ] == [
        "encoder",
        "scaler",
        "feature_names",
    ]

    assert [
        field.name
        for field in fields(
            UNSWStandardizedDataSplits
        )
    ] == [
        "normal_fit",
        "normal_calibration",
        "test",
        "preprocessor",
    ]

    assert callable(
        split_unsw_normal_calibration_test
    )
    assert callable(standardize_unsw_splits)

def _curated_partition(
    identifiers: list[int],
    labels: list[int],
) -> pd.DataFrame:
    row_count = len(identifiers)

    frame = pd.DataFrame(
        {
            column: np.arange(
                1,
                row_count + 1,
                dtype=np.float64,
            )
            for column in UNSW_CURATED_COLUMNS
        },
        columns=UNSW_CURATED_COLUMNS,
    )

    frame["id"] = np.asarray(
        identifiers,
        dtype=np.int64,
    )
    frame["proto"] = [
        "tcp" if index % 2 == 0 else "udp"
        for index in range(row_count)
    ]
    frame["service"] = [
        "http" if index % 2 == 0 else "dns"
        for index in range(row_count)
    ]
    frame["state"] = [
        "FIN" if index % 2 == 0 else "CON"
        for index in range(row_count)
    ]
    frame["label"] = np.asarray(
        labels,
        dtype=np.int64,
    )
    frame["attack_cat"] = [
        "Normal" if label == 0 else "Generic"
        for label in labels
    ]

    return frame


def _split_fixture_data() -> UNSWNB15Data:
    training = _curated_partition(
        list(range(1, 13)),
        [0] * 8 + [1] * 4,
    )
    testing = _curated_partition(
        [1, 2, 3, 4],
        [0, 1, 0, 1],
    )

    descriptions = pd.DataFrame(
        {
            "No.": [1],
            "Name": ["srcip"],
            "Type ": ["nominal"],
            "Description": [
                "Source IP address"
            ],
        }
    )

    return UNSWNB15Data(
        training=training,
        testing=testing,
        feature_descriptions=descriptions,
    )


def test_creates_normal_only_development_splits() -> None:
    data = _split_fixture_data()

    splits = (
        split_unsw_normal_calibration_test(
            data,
            normal_fit_fraction=0.75,
            random_seed=42,
        )
    )

    assert splits.normal_fit.shape[0] == 6
    assert (
        splits.normal_calibration.shape[0]
        == 2
    )
    assert splits.test.shape[0] == 4

    assert set(
        splits.normal_fit["label"]
    ) == {0}
    assert set(
        splits.normal_calibration["label"]
    ) == {0}

    fit_ids = set(splits.normal_fit["id"])
    calibration_ids = set(
        splits.normal_calibration["id"]
    )

    assert fit_ids.isdisjoint(
        calibration_ids
    )
    assert (
        fit_ids | calibration_ids
    ) == set(range(1, 9))

    assert list(splits.normal_fit["id"]) == (
        sorted(splits.normal_fit["id"])
    )
    assert list(
        splits.normal_calibration["id"]
    ) == sorted(
        splits.normal_calibration["id"]
    )

    pd.testing.assert_frame_equal(
        splits.test,
        data.testing,
    )


def test_unsw_split_is_deterministic() -> None:
    data = _split_fixture_data()

    first = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )
    second = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )
    different_seed = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=43,
        )
    )

    pd.testing.assert_frame_equal(
        first.normal_fit,
        second.normal_fit,
    )
    pd.testing.assert_frame_equal(
        first.normal_calibration,
        second.normal_calibration,
    )
    pd.testing.assert_frame_equal(
        first.test,
        second.test,
    )

    assert not first.normal_fit.equals(
        different_seed.normal_fit
    )


def test_unsw_split_does_not_mutate_data() -> None:
    data = _split_fixture_data()

    training_before = data.training.copy(
        deep=True
    )
    testing_before = data.testing.copy(
        deep=True
    )

    splits = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )

    pd.testing.assert_frame_equal(
        data.training,
        training_before,
    )
    pd.testing.assert_frame_equal(
        data.testing,
        testing_before,
    )

    assert splits.normal_fit is not (
        data.training
    )
    assert splits.test is not data.testing

def _numeric_model_columns() -> tuple[str, ...]:
    return tuple(
        column
        for column in UNSW_CURATED_COLUMNS
        if column
        not in (
            "id",
            "label",
            "attack_cat",
            "proto",
            "service",
            "state",
        )
    )


def test_standardizes_combined_model_matrix() -> None:
    data = _split_fixture_data()
    raw_splits = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )

    standardized = standardize_unsw_splits(
        raw_splits
    )

    assert isinstance(
        standardized,
        UNSWStandardizedDataSplits,
    )
    assert isinstance(
        standardized.preprocessor,
        UNSWPreprocessor,
    )

    assert standardized.normal_fit.shape[0] == 6
    assert (
        standardized.normal_calibration.shape[0]
        == 2
    )
    assert standardized.test.shape[0] == 4

    expected_feature_count = (
        len(_numeric_model_columns())
        + sum(
            len(categories)
            for categories
            in standardized
            .preprocessor
            .encoder
            .categories_
        )
    )

    assert expected_feature_count == 45

    for frame in (
        standardized.normal_fit,
        standardized.normal_calibration,
        standardized.test,
    ):
        assert frame.shape[1] == (
            expected_feature_count
        )
        assert tuple(frame.columns) == (
            standardized
            .preprocessor
            .feature_names
        )
        assert frame.index.name == "flow_id"
        assert all(
            dtype == np.dtype("float64")
            for dtype in frame.dtypes
        )
        assert np.isfinite(
            frame.to_numpy()
        ).all()

    np.testing.assert_allclose(
        standardized.normal_fit.mean(
            axis=0
        ).to_numpy(),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        standardized.normal_fit.std(
            axis=0,
            ddof=0,
        ).to_numpy(),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    assert (
        standardized
        .preprocessor
        .encoder
        .handle_unknown
        == "ignore"
    )
    assert (
        standardized
        .preprocessor
        .scaler
        .n_samples_seen_
        == 6
    )

    assert all(
        identifier.startswith(
            "unsw_training:"
        )
        for identifier
        in standardized.normal_fit.index
    )
    assert all(
        identifier.startswith(
            "unsw_testing:"
        )
        for identifier
        in standardized.test.index
    )


def test_calibration_and_test_do_not_change_preprocessor() -> None:
    data = _split_fixture_data()
    raw_splits = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )

    modified_calibration = (
        raw_splits
        .normal_calibration
        .copy(deep=True)
    )
    modified_test = raw_splits.test.copy(
        deep=True
    )

    numeric_columns = (
        _numeric_model_columns()
    )

    modified_calibration.loc[
        :,
        numeric_columns,
    ] += 1.0e6
    modified_test.loc[
        :,
        numeric_columns,
    ] -= 1.0e6

    modified_calibration.loc[
        :,
        ["proto", "service", "state"],
    ] = [
        "unseen-proto",
        "unseen-service",
        "unseen-state",
    ]
    modified_test.loc[
        :,
        ["proto", "service", "state"],
    ] = [
        "other-proto",
        "other-service",
        "other-state",
    ]

    modified_splits = UNSWRawDataSplits(
        normal_fit=raw_splits.normal_fit.copy(
            deep=True
        ),
        normal_calibration=(
            modified_calibration
        ),
        test=modified_test,
    )

    baseline = standardize_unsw_splits(
        raw_splits
    )
    modified = standardize_unsw_splits(
        modified_splits
    )

    assert (
        baseline.preprocessor.feature_names
        == modified.preprocessor.feature_names
    )

    for baseline_categories, modified_categories in zip(
        baseline.preprocessor.encoder.categories_,
        modified.preprocessor.encoder.categories_,
        strict=True,
    ):
        np.testing.assert_array_equal(
            baseline_categories,
            modified_categories,
        )

    np.testing.assert_array_equal(
        baseline.preprocessor.scaler.mean_,
        modified.preprocessor.scaler.mean_,
    )
    np.testing.assert_array_equal(
        baseline.preprocessor.scaler.scale_,
        modified.preprocessor.scaler.scale_,
    )
    pd.testing.assert_frame_equal(
        baseline.normal_fit,
        modified.normal_fit,
    )

    assert np.isfinite(
        modified.normal_calibration.to_numpy()
    ).all()
    assert np.isfinite(
        modified.test.to_numpy()
    ).all()


def test_standardization_does_not_mutate_raw_splits() -> None:
    data = _split_fixture_data()
    raw_splits = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )

    fit_before = raw_splits.normal_fit.copy(
        deep=True
    )
    calibration_before = (
        raw_splits
        .normal_calibration
        .copy(deep=True)
    )
    test_before = raw_splits.test.copy(
        deep=True
    )

    standardize_unsw_splits(raw_splits)

    pd.testing.assert_frame_equal(
        raw_splits.normal_fit,
        fit_before,
    )
    pd.testing.assert_frame_equal(
        raw_splits.normal_calibration,
        calibration_before,
    )
    pd.testing.assert_frame_equal(
        raw_splits.test,
        test_before,
    )

def test_builds_and_writes_preprocessing_evidence(
    tmp_path: Path,
) -> None:
    data = _split_fixture_data()
    raw_splits = (
        split_unsw_normal_calibration_test(
            data,
            random_seed=42,
        )
    )
    standardized = standardize_unsw_splits(
        raw_splits
    )

    evidence = (
        build_unsw_preprocessing_evidence(
            raw_splits,
            standardized,
        )
    )

    assert evidence["dataset"] == "UNSW-NB15"
    assert evidence["phase"] == 7
    assert evidence["status"] == "passed"

    assert evidence["partitions"][
        "normal_fit"
    ]["observations"] == 6
    assert evidence["partitions"][
        "normal_calibration"
    ]["observations"] == 2
    assert evidence["partitions"]["test"][
        "observations"
    ] == 4

    assert evidence["features"][
        "numeric"
    ] == 39
    assert evidence["features"][
        "categorical_inputs"
    ] == 3
    assert evidence["features"][
        "encoded_categorical"
    ] == 6
    assert evidence["features"][
        "model_features"
    ] == 45

    assert evidence["guards"] == {
        "training_attacks_excluded": True,
        "encoder_fit_normal_only": True,
        "scaler_fit_normal_only": True,
        "test_labels_used_for_fitting": False,
        "pca_fitted": False,
        "threshold_calibrated": False,
    }

    assert evidence["standardization"][
        "all_values_finite"
    ] is True

    assert json.dumps(
        evidence,
        sort_keys=True,
    )

    output_path = (
        tmp_path
        / "reports"
        / "validation"
        / "preprocessing.json"
    )

    written_path = (
        write_unsw_preprocessing_evidence(
            raw_splits,
            standardized,
            output_path=output_path,
        )
    )

    assert written_path == output_path
    assert output_path.is_file()
    assert output_path.read_bytes().endswith(
        b"\n"
    )
    assert json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    ) == evidence

    first_bytes = output_path.read_bytes()

    write_unsw_preprocessing_evidence(
        raw_splits,
        standardized,
        output_path=output_path,
    )

    assert output_path.read_bytes() == (
        first_bytes
    )
