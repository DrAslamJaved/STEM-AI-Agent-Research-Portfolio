"""Tests for UNSW-NB15 raw-data acquisition and validation."""

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyber_pca.unsw_data import (
    UNSWNB15Data,
    UNSWNB15Paths,
    UNSW_CURATED_COLUMNS,
    build_unsw_nb15_manifest,
    load_unsw_nb15,
    resolve_unsw_nb15_paths,
    validate_unsw_nb15,
    write_unsw_nb15_manifest,
)


def test_unsw_data_public_interface() -> None:
    assert is_dataclass(UNSWNB15Paths)
    assert is_dataclass(UNSWNB15Data)

    assert [
        field.name
        for field in fields(UNSWNB15Paths)
    ] == [
        "training",
        "testing",
        "feature_descriptions",
    ]

    assert [
        field.name
        for field in fields(UNSWNB15Data)
    ] == [
        "training",
        "testing",
        "feature_descriptions",
    ]

    assert callable(resolve_unsw_nb15_paths)


def test_resolves_official_raw_paths(
    tmp_path: Path,
) -> None:
    paths = resolve_unsw_nb15_paths(
        tmp_path
    )

    assert paths.training == (
        tmp_path
        / "UNSW_NB15_training-set.csv"
    )
    assert paths.testing == (
        tmp_path
        / "UNSW_NB15_testing-set.csv"
    )
    assert paths.feature_descriptions == (
        tmp_path
        / "NUSW-NB15_features.csv"
    )

    assert not paths.training.exists()
    assert not paths.testing.exists()
    assert not paths.feature_descriptions.exists()

def _write_small_raw_files(
    raw_directory: Path,
) -> tuple[
    UNSWNB15Paths,
    pd.DataFrame,
    pd.DataFrame,
]:
    paths = resolve_unsw_nb15_paths(
        raw_directory
    )

    training = pd.DataFrame(
        {
            "id": [1, 2],
            "dur": [0.1, 0.2],
            "proto": ["tcp", "udp"],
            "service": ["http", "-"],
            "state": ["FIN", "CON"],
            "attack_cat": [
                "Normal",
                "Generic",
            ],
            "label": [0, 1],
        }
    )

    testing = pd.DataFrame(
        {
            "id": [1],
            "dur": [0.3],
            "proto": ["tcp"],
            "service": ["http"],
            "state": ["FIN"],
            "attack_cat": ["Normal"],
            "label": [0],
        }
    )

    training.to_csv(
        paths.training,
        index=False,
        encoding="utf-8",
    )
    testing.to_csv(
        paths.testing,
        index=False,
        encoding="utf-8",
    )

    descriptor_text = (
        "No.,Name,Type ,Description\n"
        "1,dur,float,Connection?s duration\n"
    )

    paths.feature_descriptions.write_text(
        descriptor_text,
        encoding="cp1252",
    )

    return paths, training, testing


def test_loads_curated_and_descriptor_encodings(
    tmp_path: Path,
) -> None:
    (
        paths,
        expected_training,
        expected_testing,
    ) = _write_small_raw_files(tmp_path)

    loaded = load_unsw_nb15(paths)

    assert isinstance(loaded, UNSWNB15Data)

    pd.testing.assert_frame_equal(
        loaded.training,
        expected_training,
    )
    pd.testing.assert_frame_equal(
        loaded.testing,
        expected_testing,
    )

    assert loaded.feature_descriptions.shape == (
        1,
        4,
    )
    assert (
        loaded.feature_descriptions.loc[
            0,
            "Description",
        ]
        == "Connection?s duration"
    )


def test_load_accepts_raw_directory(
    tmp_path: Path,
) -> None:
    (
        _,
        expected_training,
        expected_testing,
    ) = _write_small_raw_files(tmp_path)

    loaded = load_unsw_nb15(tmp_path)

    pd.testing.assert_frame_equal(
        loaded.training,
        expected_training,
    )
    pd.testing.assert_frame_equal(
        loaded.testing,
        expected_testing,
    )

def _small_curated_frame() -> pd.DataFrame:
    numeric_defaults = {
        column: [1.0, 2.0]
        for column in UNSW_CURATED_COLUMNS
    }

    frame = pd.DataFrame(
        numeric_defaults,
        columns=UNSW_CURATED_COLUMNS,
    )

    frame["id"] = np.asarray(
        [1, 2],
        dtype=np.int64,
    )
    frame["proto"] = ["tcp", "udp"]
    frame["service"] = ["http", "-"]
    frame["state"] = ["FIN", "CON"]
    frame["attack_cat"] = [
        "Normal",
        "Generic",
    ]
    frame["label"] = np.asarray(
        [0, 1],
        dtype=np.int64,
    )

    return frame


def _small_unsw_data() -> UNSWNB15Data:
    feature_descriptions = pd.DataFrame(
        {
            "No.": [1, 49],
            "Name": ["srcip", "Label"],
            "Type ": ["nominal", "binary"],
            "Description": [
                "Source IP address",
                "0 normal; 1 attack",
            ],
        }
    )

    return UNSWNB15Data(
        training=_small_curated_frame(),
        testing=_small_curated_frame(),
        feature_descriptions=(
            feature_descriptions
        ),
    )


def test_validates_verified_curated_schema() -> None:
    data = _small_unsw_data()

    validated = validate_unsw_nb15(
        data,
        expected_training_rows=2,
        expected_testing_rows=2,
        expected_feature_description_rows=2,
    )

    assert validated is data


def test_validation_rejects_schema_mismatch() -> None:
    data = _small_unsw_data()

    invalid_data = UNSWNB15Data(
        training=data.training.drop(
            columns=["dur"]
        ),
        testing=data.testing,
        feature_descriptions=(
            data.feature_descriptions
        ),
    )

    with pytest.raises(
        ValueError,
        match="training columns",
    ):
        validate_unsw_nb15(
            invalid_data,
            expected_training_rows=2,
            expected_testing_rows=2,
            expected_feature_description_rows=2,
        )


def test_validation_does_not_mutate_data() -> None:
    data = _small_unsw_data()

    training_before = data.training.copy(
        deep=True
    )
    testing_before = data.testing.copy(
        deep=True
    )
    descriptions_before = (
        data.feature_descriptions.copy(
            deep=True
        )
    )

    validate_unsw_nb15(
        data,
        expected_training_rows=2,
        expected_testing_rows=2,
        expected_feature_description_rows=2,
    )

    pd.testing.assert_frame_equal(
        data.training,
        training_before,
    )
    pd.testing.assert_frame_equal(
        data.testing,
        testing_before,
    )
    pd.testing.assert_frame_equal(
        data.feature_descriptions,
        descriptions_before,
    )

def _write_schema_fixture(
    raw_directory: Path,
) -> tuple[
    UNSWNB15Paths,
    UNSWNB15Data,
]:
    paths = resolve_unsw_nb15_paths(
        raw_directory
    )
    data = _small_unsw_data()

    data.training.to_csv(
        paths.training,
        index=False,
        encoding="utf-8",
    )
    data.testing.to_csv(
        paths.testing,
        index=False,
        encoding="utf-8",
    )
    data.feature_descriptions.to_csv(
        paths.feature_descriptions,
        index=False,
        encoding="cp1252",
    )

    return paths, data


def test_builds_deterministic_json_manifest(
    tmp_path: Path,
) -> None:
    paths, data = _write_schema_fixture(
        tmp_path
    )

    manifest = build_unsw_nb15_manifest(
        paths,
        data,
        expected_training_rows=2,
        expected_testing_rows=2,
        expected_feature_description_rows=2,
    )

    assert manifest["dataset"] == {
        "name": "UNSW-NB15",
        "source_page": (
            "https://research.unsw.edu.au/"
            "projects/unsw-nb15-dataset"
        ),
        "acquisition_method": (
            "manual_official_download"
        ),
        "academic_use": True,
        "raw_files_immutable": True,
    }

    assert manifest["validation"] == {
        "status": "passed",
        "curated_schema_columns": 45,
        "feature_description_rows": 2,
        "identifier_scope": "partition_local",
        "record_key": [
            "source_partition",
            "id",
        ],
    }

    training_evidence = (
        manifest["files"]["training"]
    )

    assert training_evidence["filename"] == (
        "UNSW_NB15_training-set.csv"
    )
    assert training_evidence["encoding"] == (
        "utf-8"
    )
    assert training_evidence["rows"] == 2
    assert training_evidence["columns"] == 45
    assert training_evidence["label_counts"] == {
        "0": 1,
        "1": 1,
    }
    assert training_evidence[
        "attack_category_counts"
    ] == {
        "Generic": 1,
        "Normal": 1,
    }

    training_bytes = paths.training.read_bytes()

    assert training_evidence["bytes"] == len(
        training_bytes
    )
    assert training_evidence["sha256"] == (
        sha256(training_bytes).hexdigest()
    )

    feature_evidence = manifest["files"][
        "feature_descriptions"
    ]

    assert feature_evidence["filename"] == (
        "NUSW-NB15_features.csv"
    )
    assert feature_evidence["encoding"] == (
        "cp1252"
    )
    assert feature_evidence["rows"] == 2
    assert feature_evidence["columns"] == 4

    serialized = json.dumps(
        manifest,
        sort_keys=True,
    )

    assert serialized
    assert (
        manifest
        == build_unsw_nb15_manifest(
            paths,
            data,
            expected_training_rows=2,
            expected_testing_rows=2,
            expected_feature_description_rows=2,
        )
    )

def test_writes_deterministic_manifest(
    tmp_path: Path,
) -> None:
    paths, data = _write_schema_fixture(
        tmp_path
    )

    output_path = (
        tmp_path
        / "reports"
        / "validation"
        / "manifest.json"
    )

    written_path = write_unsw_nb15_manifest(
        paths,
        data,
        output_path=output_path,
        expected_training_rows=2,
        expected_testing_rows=2,
        expected_feature_description_rows=2,
    )

    assert written_path == output_path
    assert output_path.is_file()
    assert output_path.read_bytes().endswith(
        b"\n"
    )

    expected_manifest = (
        build_unsw_nb15_manifest(
            paths,
            data,
            expected_training_rows=2,
            expected_testing_rows=2,
            expected_feature_description_rows=2,
        )
    )

    assert json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    ) == expected_manifest

    first_bytes = output_path.read_bytes()

    repeated_path = write_unsw_nb15_manifest(
        paths,
        data,
        output_path=output_path,
        expected_training_rows=2,
        expected_testing_rows=2,
        expected_feature_description_rows=2,
    )

    assert repeated_path == output_path
    assert output_path.read_bytes() == (
        first_bytes
    )
