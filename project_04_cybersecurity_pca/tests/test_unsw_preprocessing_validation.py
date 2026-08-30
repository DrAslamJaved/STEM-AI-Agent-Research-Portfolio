"""Boundary tests for UNSW-NB15 preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cyber_pca.unsw_data import (
    UNSWNB15Data,
    UNSW_CURATED_COLUMNS,
)
from cyber_pca.unsw_preprocessing import (
    UNSWRawDataSplits,
    UNSWStandardizedDataSplits,
    build_unsw_preprocessing_evidence,
    split_unsw_normal_calibration_test,
    standardize_unsw_splits,
    write_unsw_preprocessing_evidence,
)


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


def _data(
    *,
    training: object | None = None,
    testing: object | None = None,
) -> UNSWNB15Data:
    return UNSWNB15Data(
        training=(
            _curated_partition(
                list(range(1, 13)),
                [0] * 8 + [1] * 4,
            )
            if training is None
            else training
        ),
        testing=(
            _curated_partition(
                [1, 2, 3, 4],
                [0, 1, 0, 1],
            )
            if testing is None
            else testing
        ),
        feature_descriptions=pd.DataFrame(),
    )


def _raw_splits() -> UNSWRawDataSplits:
    return split_unsw_normal_calibration_test(
        _data(),
        random_seed=42,
    )


def _standardized_pair() -> tuple[
    UNSWRawDataSplits,
    UNSWStandardizedDataSplits,
]:
    raw_splits = _raw_splits()

    return (
        raw_splits,
        standardize_unsw_splits(raw_splits),
    )


def _replace_raw(
    source: UNSWRawDataSplits,
    *,
    normal_fit: object | None = None,
    normal_calibration: object | None = None,
    test: object | None = None,
) -> UNSWRawDataSplits:
    return UNSWRawDataSplits(
        normal_fit=(
            source.normal_fit
            if normal_fit is None
            else normal_fit
        ),
        normal_calibration=(
            source.normal_calibration
            if normal_calibration is None
            else normal_calibration
        ),
        test=(
            source.test
            if test is None
            else test
        ),
    )


def _replace_standardized(
    source: UNSWStandardizedDataSplits,
    *,
    normal_fit: object | None = None,
    normal_calibration: object | None = None,
    test: object | None = None,
) -> UNSWStandardizedDataSplits:
    return UNSWStandardizedDataSplits(
        normal_fit=(
            source.normal_fit
            if normal_fit is None
            else normal_fit
        ),
        normal_calibration=(
            source.normal_calibration
            if normal_calibration is None
            else normal_calibration
        ),
        test=(
            source.test
            if test is None
            else test
        ),
        preprocessor=source.preprocessor,
    )


def test_split_rejects_invalid_data_type() -> None:
    with pytest.raises(TypeError):
        split_unsw_normal_calibration_test(
            object()
        )


@pytest.mark.parametrize(
    "invalid_fraction",
    [
        True,
        "0.75",
        None,
    ],
)
def test_split_rejects_nonnumeric_fraction(
    invalid_fraction: object,
) -> None:
    with pytest.raises(TypeError):
        split_unsw_normal_calibration_test(
            _data(),
            normal_fit_fraction=(
                invalid_fraction
            ),
        )


@pytest.mark.parametrize(
    "invalid_fraction",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        np.nan,
        np.inf,
    ],
)
def test_split_rejects_invalid_fraction_range(
    invalid_fraction: float,
) -> None:
    with pytest.raises(ValueError):
        split_unsw_normal_calibration_test(
            _data(),
            normal_fit_fraction=(
                invalid_fraction
            ),
        )


@pytest.mark.parametrize(
    "invalid_seed",
    [
        True,
        1.5,
        "42",
    ],
)
def test_split_rejects_noninteger_seed(
    invalid_seed: object,
) -> None:
    with pytest.raises(TypeError):
        split_unsw_normal_calibration_test(
            _data(),
            random_seed=invalid_seed,
        )


def test_split_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        split_unsw_normal_calibration_test(
            _data(),
            random_seed=-1,
        )


def test_split_rejects_non_dataframe_partition() -> None:
    with pytest.raises(
        TypeError,
        match="training",
    ):
        split_unsw_normal_calibration_test(
            _data(training=object())
        )


def test_split_rejects_empty_partition() -> None:
    training = _curated_partition(
        [1, 2],
        [0, 1],
    ).iloc[:0].copy()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        split_unsw_normal_calibration_test(
            _data(training=training)
        )


def test_split_rejects_schema_mismatch() -> None:
    training = _data().training.drop(
        columns=["dur"]
    )

    with pytest.raises(
        ValueError,
        match="columns",
    ):
        split_unsw_normal_calibration_test(
            _data(training=training)
        )


def test_split_rejects_missing_id() -> None:
    training = _data().training.copy()
    training.loc[0, "id"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing IDs",
    ):
        split_unsw_normal_calibration_test(
            _data(training=training)
        )


def test_split_rejects_duplicate_id() -> None:
    training = _data().training.copy()
    training.loc[1, "id"] = 1

    with pytest.raises(
        ValueError,
        match="duplicate IDs",
    ):
        split_unsw_normal_calibration_test(
            _data(training=training)
        )


def test_split_requires_two_normal_rows() -> None:
    training = _curated_partition(
        [1, 2],
        [0, 1],
    )

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        split_unsw_normal_calibration_test(
            _data(training=training)
        )


def test_standardization_rejects_invalid_splits() -> None:
    with pytest.raises(TypeError):
        standardize_unsw_splits(object())


def test_standardization_rejects_non_dataframe() -> None:
    raw_splits = _raw_splits()

    with pytest.raises(
        TypeError,
        match="normal_fit",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=object(),
            )
        )


def test_standardization_rejects_empty_frame() -> None:
    raw_splits = _raw_splits()

    empty_fit = (
        raw_splits.normal_fit.iloc[:0]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=empty_fit,
            )
        )


def test_standardization_rejects_schema() -> None:
    raw_splits = _raw_splits()

    invalid_fit = (
        raw_splits.normal_fit.drop(
            columns=["dur"]
        )
    )

    with pytest.raises(
        ValueError,
        match="columns",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_missing_id() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit.loc[
        invalid_fit.index[0],
        "id",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="missing IDs",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_duplicate_id() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit.loc[
        invalid_fit.index[1],
        "id",
    ] = invalid_fit.loc[
        invalid_fit.index[0],
        "id",
    ]

    with pytest.raises(
        ValueError,
        match="duplicate IDs",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_missing_model_value() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit.loc[
        invalid_fit.index[0],
        "dur",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="missing model values",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_nonnumeric_value() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit["dur"] = invalid_fit[
        "dur"
    ].astype(object)
    invalid_fit.loc[
        invalid_fit.index[0],
        "dur",
    ] = "invalid"

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_nonfinite_value() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit.loc[
        invalid_fit.index[0],
        "dur",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="nonfinite",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_attack_in_fit() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit.loc[
        invalid_fit.index[0],
        "label",
    ] = 1

    with pytest.raises(
        ValueError,
        match="normal_fit",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_standardization_rejects_attack_in_calibration() -> None:
    raw_splits = _raw_splits()
    invalid_calibration = (
        raw_splits
        .normal_calibration
        .copy()
    )
    invalid_calibration.loc[
        invalid_calibration.index[0],
        "label",
    ] = 1

    with pytest.raises(
        ValueError,
        match="normal_calibration",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_calibration=(
                    invalid_calibration
                ),
            )
        )


def test_standardization_rejects_overlapping_ids() -> None:
    raw_splits = _raw_splits()
    invalid_calibration = (
        raw_splits
        .normal_calibration
        .copy()
    )
    invalid_calibration.loc[
        invalid_calibration.index[0],
        "id",
    ] = raw_splits.normal_fit.loc[
        raw_splits.normal_fit.index[0],
        "id",
    ]

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_calibration=(
                    invalid_calibration
                ),
            )
        )


def test_standardization_rejects_zero_variance() -> None:
    raw_splits = _raw_splits()
    invalid_fit = (
        raw_splits.normal_fit.copy()
    )
    invalid_fit["dur"] = 1.0

    with pytest.raises(
        ValueError,
        match="zero-variance",
    ):
        standardize_unsw_splits(
            _replace_raw(
                raw_splits,
                normal_fit=invalid_fit,
            )
        )


def test_evidence_rejects_invalid_raw_type() -> None:
    _, standardized = _standardized_pair()

    with pytest.raises(TypeError):
        build_unsw_preprocessing_evidence(
            object(),
            standardized,
        )


def test_evidence_rejects_invalid_standardized_type() -> None:
    raw_splits = _raw_splits()

    with pytest.raises(TypeError):
        build_unsw_preprocessing_evidence(
            raw_splits,
            object(),
        )


def test_evidence_rejects_invalid_raw_partition() -> None:
    raw_splits, standardized = (
        _standardized_pair()
    )

    with pytest.raises(
        TypeError,
        match="raw partition",
    ):
        build_unsw_preprocessing_evidence(
            _replace_raw(
                raw_splits,
                normal_fit=object(),
            ),
            standardized,
        )


def test_evidence_rejects_invalid_model_partition() -> None:
    raw_splits, standardized = (
        _standardized_pair()
    )

    with pytest.raises(
        TypeError,
        match="standardized partition",
    ):
        build_unsw_preprocessing_evidence(
            raw_splits,
            _replace_standardized(
                standardized,
                normal_fit=object(),
            ),
        )


def test_evidence_rejects_row_mismatch() -> None:
    raw_splits, standardized = (
        _standardized_pair()
    )

    shortened = (
        standardized.normal_fit.iloc[:-1]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="row counts differ",
    ):
        build_unsw_preprocessing_evidence(
            raw_splits,
            _replace_standardized(
                standardized,
                normal_fit=shortened,
            ),
        )


def test_evidence_rejects_column_mismatch() -> None:
    raw_splits, standardized = (
        _standardized_pair()
    )

    invalid_columns = (
        standardized.normal_fit.drop(
            columns=[
                standardized
                .normal_fit
                .columns[0]
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="feature_names",
    ):
        build_unsw_preprocessing_evidence(
            raw_splits,
            _replace_standardized(
                standardized,
                normal_fit=invalid_columns,
            ),
        )


def test_evidence_rejects_nonfinite_values() -> None:
    raw_splits, standardized = (
        _standardized_pair()
    )

    invalid_fit = (
        standardized.normal_fit.copy()
    )
    invalid_fit.iloc[0, 0] = np.inf

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        build_unsw_preprocessing_evidence(
            raw_splits,
            _replace_standardized(
                standardized,
                normal_fit=invalid_fit,
            ),
        )


def test_evidence_writer_rejects_output_type() -> None:
    raw_splits, standardized = (
        _standardized_pair()
    )

    with pytest.raises(TypeError):
        write_unsw_preprocessing_evidence(
            raw_splits,
            standardized,
            output_path=object(),
        )
