"""Boundary tests for UNSW-NB15 data validation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyber_pca.unsw_data import (
    UNSWNB15Data,
    UNSW_CURATED_COLUMNS,
    build_unsw_nb15_manifest,
    load_unsw_nb15,
    resolve_unsw_nb15_paths,
    validate_unsw_nb15,
    write_unsw_nb15_manifest,
)


def _curated_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            column: [1.0, 2.0]
            for column in UNSW_CURATED_COLUMNS
        },
        columns=UNSW_CURATED_COLUMNS,
    )

    frame["id"] = np.asarray(
        [1, 2],
        dtype=np.int64,
    )
    frame["proto"] = ["tcp", "udp"]
    frame["service"] = ["http", "dns"]
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


def _descriptions() -> pd.DataFrame:
    return pd.DataFrame(
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


def _data(
    *,
    training: object | None = None,
    testing: object | None = None,
    descriptions: object | None = None,
) -> UNSWNB15Data:
    return UNSWNB15Data(
        training=(
            _curated_frame()
            if training is None
            else training
        ),
        testing=(
            _curated_frame()
            if testing is None
            else testing
        ),
        feature_descriptions=(
            _descriptions()
            if descriptions is None
            else descriptions
        ),
    )


def _validate(data: object) -> UNSWNB15Data:
    return validate_unsw_nb15(
        data,
        expected_training_rows=2,
        expected_testing_rows=2,
        expected_feature_description_rows=2,
    )


@pytest.mark.parametrize(
    "invalid_directory",
    [
        None,
        1,
        True,
        object(),
    ],
)
def test_path_resolution_rejects_invalid_types(
    invalid_directory: object,
) -> None:
    with pytest.raises(TypeError):
        resolve_unsw_nb15_paths(
            invalid_directory
        )


def test_loading_rejects_missing_raw_files(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="training file",
    ):
        load_unsw_nb15(tmp_path)


@pytest.mark.parametrize(
    (
        "argument",
        "invalid_value",
        "expected_exception",
    ),
    [
        (
            "expected_training_rows",
            True,
            TypeError,
        ),
        (
            "expected_testing_rows",
            2.5,
            TypeError,
        ),
        (
            "expected_feature_description_rows",
            "2",
            TypeError,
        ),
        (
            "expected_training_rows",
            0,
            ValueError,
        ),
        (
            "expected_testing_rows",
            -1,
            ValueError,
        ),
    ],
)
def test_validation_rejects_invalid_expected_counts(
    argument: str,
    invalid_value: object,
    expected_exception: type[Exception],
) -> None:
    keyword_arguments = {
        "expected_training_rows": 2,
        "expected_testing_rows": 2,
        "expected_feature_description_rows": 2,
    }
    keyword_arguments[argument] = invalid_value

    with pytest.raises(expected_exception):
        validate_unsw_nb15(
            _data(),
            **keyword_arguments,
        )


def test_validation_rejects_invalid_data_type() -> None:
    with pytest.raises(TypeError):
        _validate(object())


def test_validation_rejects_non_dataframe_partition() -> None:
    with pytest.raises(
        TypeError,
        match="training",
    ):
        _validate(
            _data(training=object())
        )


def test_validation_rejects_wrong_row_count() -> None:
    training = _curated_frame().iloc[
        :1
    ].copy()

    with pytest.raises(
        ValueError,
        match="training row count",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_duplicate_rows() -> None:
    training = (
        _curated_frame()
        .iloc[[0, 0]]
        .reset_index(drop=True)
    )

    with pytest.raises(
        ValueError,
        match="duplicate rows",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_missing_values() -> None:
    training = _curated_frame()
    training.loc[0, "dur"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_duplicate_ids() -> None:
    training = _curated_frame()
    training.loc[1, "id"] = 1

    with pytest.raises(
        ValueError,
        match="duplicate IDs",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_nonnumeric_values() -> None:
    training = _curated_frame()
    training["dur"] = training[
        "dur"
    ].astype(object)
    training.loc[0, "dur"] = "invalid"

    with pytest.raises(
        TypeError,
        match="numeric columns",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_nonnumeric_id_dtype() -> None:
    training = _curated_frame()
    training["id"] = ["1", "2"]

    with pytest.raises(
        TypeError,
        match="IDs must be numeric",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_infinite_values() -> None:
    training = _curated_frame()
    training.loc[0, "dur"] = np.inf

    with pytest.raises(
        ValueError,
        match="nonfinite",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_nonbinary_support() -> None:
    training = _curated_frame()
    training["label"] = [0, 0]
    training["attack_cat"] = [
        "Normal",
        "Normal",
    ]

    with pytest.raises(
        ValueError,
        match="labels",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_label_category_mismatch() -> None:
    training = _curated_frame()
    training.loc[0, "attack_cat"] = (
        "Generic"
    )

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        _validate(
            _data(training=training)
        )


def test_validation_rejects_non_dataframe_descriptions() -> None:
    with pytest.raises(
        TypeError,
        match="feature_descriptions",
    ):
        _validate(
            _data(descriptions=object())
        )


def test_validation_rejects_description_row_count() -> None:
    descriptions = _descriptions().iloc[
        :1
    ].copy()

    with pytest.raises(
        ValueError,
        match="row count",
    ):
        _validate(
            _data(
                descriptions=descriptions
            )
        )


def test_validation_rejects_description_columns() -> None:
    descriptions = _descriptions().rename(
        columns={"Type ": "Type"}
    )

    with pytest.raises(
        ValueError,
        match="columns",
    ):
        _validate(
            _data(
                descriptions=descriptions
            )
        )


def test_validation_rejects_missing_description() -> None:
    descriptions = _descriptions()
    descriptions.loc[
        0,
        "Description",
    ] = None

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        _validate(
            _data(
                descriptions=descriptions
            )
        )


def test_manifest_rejects_invalid_paths_type() -> None:
    with pytest.raises(TypeError):
        build_unsw_nb15_manifest(
            object(),
            _data(),
            expected_training_rows=2,
            expected_testing_rows=2,
            expected_feature_description_rows=2,
        )


def test_manifest_writer_rejects_output_type(
    tmp_path: Path,
) -> None:
    paths = resolve_unsw_nb15_paths(
        tmp_path
    )

    with pytest.raises(TypeError):
        write_unsw_nb15_manifest(
            paths,
            _data(),
            output_path=object(),
            expected_training_rows=2,
            expected_testing_rows=2,
            expected_feature_description_rows=2,
        )
