"""Validation tests for official UNSW-NB15 reporting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

import cyber_pca.unsw_reporting as reporting
from cyber_pca.evaluation import (
    BinaryEvaluationResult,
    evaluate_binary_predictions,
)
from cyber_pca.unsw_evaluation import (
    UNSW_ATTACK_CATEGORY_ORDER,
    align_unsw_evaluation_data,
    evaluate_unsw_attack_categories,
)
from cyber_pca.unsw_experiment import (
    UNSWDetectionResult,
    run_unsw_detection,
)
from cyber_pca.unsw_preprocessing import (
    UNSWPreprocessor,
    UNSWStandardizedDataSplits,
)


FEATURE_NAMES = (
    "feature_a",
    "feature_b",
    "feature_c",
)


def _frame(
    values: object,
    identifiers: tuple[str, ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(
            values,
            dtype=np.float64,
        ),
        columns=FEATURE_NAMES,
        index=pd.Index(
            identifiers,
            name="flow_id",
        ),
    )


def _reporting_fixture(
) -> tuple[
    pd.DataFrame,
    UNSWDetectionResult,
    BinaryEvaluationResult,
    pd.DataFrame,
]:
    normal_fit = _frame(
        (
            (-1.0, -1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
        ),
        (
            "fit:1",
            "fit:2",
            "fit:3",
            "fit:4",
        ),
    )

    normal_calibration = _frame(
        (
            (0.5, 0.2, -0.3),
            (-0.5, -0.2, 0.3),
            (0.2, -0.4, 0.6),
            (-0.2, 0.4, -0.6),
        ),
        (
            "calibration:1",
            "calibration:2",
            "calibration:3",
            "calibration:4",
        ),
    )

    test_values = tuple(
        (
            value / 10.0,
            (-1.0) ** value * 0.5,
            float((value % 3) - 1),
        )
        for value in range(1, 11)
    )

    test_identifiers = tuple(
        f"unsw_testing:{value}"
        for value in range(1, 11)
    )

    test = _frame(
        test_values,
        test_identifiers,
    )

    scaler = StandardScaler().fit(
        normal_fit.to_numpy(
            dtype=np.float64,
        )
    )

    preprocessor = UNSWPreprocessor(
        encoder=OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ),
        scaler=scaler,
        feature_names=FEATURE_NAMES,
    )

    standardized = (
        UNSWStandardizedDataSplits(
            normal_fit=normal_fit,
            normal_calibration=(
                normal_calibration
            ),
            test=test,
            preprocessor=preprocessor,
        )
    )

    detection = run_unsw_detection(
        standardized
    )

    raw_test = pd.DataFrame(
        {
            "id": np.arange(
                1,
                11,
                dtype=np.int64,
            ),
            "label": np.asarray(
                [0] + [1] * 9,
                dtype=np.int8,
            ),
            "attack_cat": (
                UNSW_ATTACK_CATEGORY_ORDER
            ),
        }
    )

    evaluation = (
        align_unsw_evaluation_data(
            raw_test,
            detection
            .reconstruction_errors.test,
            detection.test_predictions,
        )
    )

    binary = evaluate_binary_predictions(
        evaluation
    )

    categories = (
        evaluate_unsw_attack_categories(
            evaluation
        )
    )

    return (
        evaluation,
        detection,
        binary,
        categories,
    )


def test_to_builtin_converts_numpy_values() -> None:
    scalar = reporting._to_builtin(
        np.int64(7)
    )

    array = reporting._to_builtin(
        np.asarray(
            (1.5, 2.5),
            dtype=np.float64,
        )
    )

    assert scalar == 7
    assert isinstance(scalar, int)
    assert array == [1.5, 2.5]
    assert all(
        isinstance(value, float)
        for value in array
    )


def test_resolver_rejects_invalid_output_root() -> None:
    with pytest.raises(
        TypeError,
        match="string or Path",
    ):
        reporting.resolve_unsw_evaluation_artifacts(
            object()
        )


@pytest.mark.parametrize(
    ("case", "exception_type", "message"),
    (
        (
            "invalid_detection",
            TypeError,
            "UNSWDetectionResult",
        ),
        (
            "invalid_binary",
            TypeError,
            "BinaryEvaluationResult",
        ),
        (
            "invalid_categories",
            TypeError,
            "pandas DataFrame",
        ),
        (
            "empty_categories",
            ValueError,
            "must not be empty",
        ),
        (
            "wrong_columns",
            ValueError,
            "columns must exactly match",
        ),
        (
            "wrong_order",
            ValueError,
            "official category order",
        ),
        (
            "category_total",
            ValueError,
            "observations do not match",
        ),
        (
            "prediction_total",
            ValueError,
            "Frozen predictions do not match",
        ),
        (
            "error_total",
            ValueError,
            "reconstruction errors do not match",
        ),
    ),
)
def test_summary_rejects_invalid_inputs(
    case: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    (
        _,
        detection,
        binary,
        categories,
    ) = _reporting_fixture()

    detection_input: object = detection
    binary_input: object = binary
    category_input: object = categories

    if case == "invalid_detection":
        detection_input = object()
    elif case == "invalid_binary":
        binary_input = object()
    elif case == "invalid_categories":
        category_input = object()
    elif case == "empty_categories":
        category_input = categories.iloc[
            0:0
        ].copy()
    elif case == "wrong_columns":
        category_input = categories.rename(
            columns={
                "attack_category": (
                    "wrong_category"
                )
            }
        )
    elif case == "wrong_order":
        category_input = (
            categories.iloc[::-1]
            .reset_index(drop=True)
        )
    elif case == "category_total":
        category_input = categories.copy(
            deep=True
        )
        category_input.loc[
            0,
            "observations",
        ] += 1
    elif case == "prediction_total":
        detection_input = replace(
            detection,
            test_predictions=(
                detection.test_predictions
                .iloc[:-1]
                .copy()
            ),
        )
    elif case == "error_total":
        errors = replace(
            detection.reconstruction_errors,
            test=(
                detection
                .reconstruction_errors.test
                .iloc[:-1]
                .copy()
            ),
        )

        detection_input = replace(
            detection,
            reconstruction_errors=errors,
        )
    else:
        raise AssertionError(case)

    with pytest.raises(
        exception_type,
        match=message,
    ):
        reporting.build_unsw_evaluation_summary(
            detection_input,
            binary_input,
            category_input,
        )


@pytest.mark.parametrize(
    ("case", "exception_type", "message"),
    (
        (
            "not_dataframe",
            TypeError,
            "pandas DataFrame",
        ),
        (
            "empty",
            ValueError,
            "must not be empty",
        ),
        (
            "wrong_columns",
            ValueError,
            "columns must exactly match",
        ),
        (
            "unnamed_index",
            ValueError,
            "index must be named flow_id",
        ),
        (
            "missing_id",
            ValueError,
            "missing flow IDs",
        ),
        (
            "duplicate_id",
            ValueError,
            "duplicate flow IDs",
        ),
        (
            "boolean_dpi",
            TypeError,
            "dpi must be an integer",
        ),
        (
            "nonnumeric_dpi",
            TypeError,
            "dpi must be an integer",
        ),
        (
            "nonpositive_dpi",
            ValueError,
            "dpi must be positive",
        ),
        (
            "row_total",
            ValueError,
            "rows do not match",
        ),
        (
            "prediction_ids",
            ValueError,
            "frozen prediction IDs",
        ),
        (
            "error_ids",
            ValueError,
            "reconstruction-error IDs",
        ),
    ),
)
def test_writer_rejects_invalid_inputs(
    case: str,
    exception_type: type[Exception],
    message: str,
    tmp_path: Path,
) -> None:
    (
        evaluation,
        detection,
        binary,
        categories,
    ) = _reporting_fixture()

    evaluation_input: object = (
        evaluation.copy(deep=True)
    )

    detection_input = detection
    dpi: object = 150

    if case == "not_dataframe":
        evaluation_input = object()
    elif case == "empty":
        evaluation_input = (
            evaluation.iloc[0:0].copy()
        )
    elif case == "wrong_columns":
        evaluation_input = (
            evaluation.rename(
                columns={
                    "scenario": (
                        "wrong_scenario"
                    )
                }
            )
        )
    elif case == "unnamed_index":
        evaluation_input.index = (
            evaluation_input.index.rename(
                None
            )
        )
    elif case == "missing_id":
        identifiers = list(
            evaluation_input.index
        )
        identifiers[0] = None
        evaluation_input.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "duplicate_id":
        identifiers = list(
            evaluation_input.index
        )
        identifiers[1] = identifiers[0]
        evaluation_input.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "boolean_dpi":
        dpi = True
    elif case == "nonnumeric_dpi":
        dpi = 150.0
    elif case == "nonpositive_dpi":
        dpi = 0
    elif case == "row_total":
        evaluation_input = (
            evaluation.iloc[:-1].copy()
        )
    elif case == "prediction_ids":
        identifiers = list(
            evaluation_input.index
        )
        identifiers[0] = (
            "unsw_testing:999"
        )
        evaluation_input.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "error_ids":
        identifiers = list(
            evaluation_input.index
        )
        identifiers[0] = (
            "unsw_testing:999"
        )
        changed_index = pd.Index(
            identifiers,
            name="flow_id",
        )

        evaluation_input.index = (
            changed_index
        )

        predictions = (
            detection.test_predictions
            .copy(deep=True)
        )
        predictions.index = changed_index

        detection_input = replace(
            detection,
            test_predictions=predictions,
        )
    else:
        raise AssertionError(case)

    with pytest.raises(
        exception_type,
        match=message,
    ):
        reporting.write_unsw_evaluation_artifacts(
            evaluation_input,
            detection_input,
            binary,
            categories,
            output_root=tmp_path,
            dpi=dpi,
        )

    assert not any(
        tmp_path.rglob("*")
    )
