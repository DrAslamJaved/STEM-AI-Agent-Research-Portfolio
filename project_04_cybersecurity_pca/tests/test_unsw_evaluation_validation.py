"""Validation tests for official UNSW-NB15 evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cyber_pca.unsw_evaluation import (
    UNSW_ATTACK_CATEGORY_ORDER,
    align_unsw_evaluation_data,
    evaluate_unsw_attack_categories,
)


def _evaluation_fixture(
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
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

    flow_ids = pd.Index(
        [
            f"unsw_testing:{value}"
            for value in range(1, 11)
        ],
        name="flow_id",
    )

    errors = pd.Series(
        np.linspace(
            0.05,
            0.95,
            num=10,
            dtype=np.float64,
        ),
        index=flow_ids,
        name="reconstruction_error",
    )

    predictions = pd.Series(
        np.asarray(
            (
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
            ),
            dtype=np.int8,
        ),
        index=flow_ids,
        name="predicted_anomaly",
    )

    return raw_test, errors, predictions


def _aligned_evaluation() -> pd.DataFrame:
    raw_test, errors, predictions = (
        _evaluation_fixture()
    )

    return align_unsw_evaluation_data(
        raw_test,
        errors,
        predictions,
    )


@pytest.mark.parametrize(
    ("case", "exception_type", "message"),
    (
        (
            "not_series",
            TypeError,
            "pandas Series",
        ),
        (
            "empty",
            ValueError,
            "must not be empty",
        ),
        (
            "wrong_name",
            ValueError,
            "must be named",
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
            "missing_value",
            ValueError,
            "missing values",
        ),
    ),
)
def test_alignment_rejects_invalid_indexed_series(
    case: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    raw_test, errors, predictions = (
        _evaluation_fixture()
    )

    invalid: object

    if case == "not_series":
        invalid = np.zeros(10)
    elif case == "empty":
        invalid = pd.Series(
            [],
            index=pd.Index(
                [],
                name="flow_id",
            ),
            name="reconstruction_error",
            dtype=np.float64,
        )
    else:
        invalid = errors.copy(deep=True)

        if case == "wrong_name":
            invalid.name = "wrong_name"
        elif case == "unnamed_index":
            invalid.index = (
                invalid.index.rename(None)
            )
        elif case == "missing_id":
            identifiers = list(
                invalid.index
            )
            identifiers[0] = None
            invalid.index = pd.Index(
                identifiers,
                name="flow_id",
            )
        elif case == "duplicate_id":
            identifiers = list(
                invalid.index
            )
            identifiers[1] = (
                identifiers[0]
            )
            invalid.index = pd.Index(
                identifiers,
                name="flow_id",
            )
        elif case == "missing_value":
            invalid.iloc[0] = np.nan
        else:
            raise AssertionError(case)

    with pytest.raises(
        exception_type,
        match=message,
    ):
        align_unsw_evaluation_data(
            raw_test,
            invalid,
            predictions,
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
            "missing_column",
            ValueError,
            "missing required columns",
        ),
        (
            "missing_id",
            ValueError,
            "evaluation columns contain missing",
        ),
        (
            "missing_label",
            ValueError,
            "evaluation columns contain missing",
        ),
        (
            "missing_category",
            ValueError,
            "evaluation columns contain missing",
        ),
        (
            "boolean_id",
            TypeError,
            "IDs must be numeric",
        ),
        (
            "nonnumeric_id",
            TypeError,
            "IDs must be numeric",
        ),
        (
            "nonfinite_id",
            ValueError,
            "IDs must be finite",
        ),
        (
            "fractional_id",
            ValueError,
            "IDs must be integers",
        ),
        (
            "nonpositive_id",
            ValueError,
            "IDs must be positive",
        ),
        (
            "duplicate_id",
            ValueError,
            "duplicate IDs",
        ),
        (
            "boolean_label",
            TypeError,
            "labels must be numeric",
        ),
        (
            "nonnumeric_label",
            TypeError,
            "labels must be numeric",
        ),
        (
            "nonfinite_label",
            ValueError,
            "labels must be finite",
        ),
        (
            "invalid_label",
            ValueError,
            "only 0 and 1",
        ),
        (
            "nonstr_category",
            TypeError,
            "categories must be strings",
        ),
        (
            "invalid_categories",
            ValueError,
            "exactly the official",
        ),
        (
            "inconsistent_category_label",
            ValueError,
            "categories are inconsistent",
        ),
    ),
)
def test_alignment_rejects_invalid_raw_test(
    case: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    raw_test, errors, predictions = (
        _evaluation_fixture()
    )

    invalid: object

    if case == "not_dataframe":
        invalid = object()
    elif case == "empty":
        invalid = raw_test.iloc[
            0:0
        ].copy()
    elif case == "missing_column":
        invalid = raw_test.drop(
            columns="label"
        )
    else:
        invalid = raw_test.copy(
            deep=True
        )

        if case == "missing_id":
            invalid["id"] = (
                invalid["id"]
                .astype(np.float64)
            )
            invalid.loc[0, "id"] = np.nan
        elif case == "missing_label":
            invalid["label"] = (
                invalid["label"]
                .astype(np.float64)
            )
            invalid.loc[0, "label"] = (
                np.nan
            )
        elif case == "missing_category":
            invalid["attack_cat"] = (
                invalid["attack_cat"]
                .astype(object)
            )
            invalid.loc[
                0,
                "attack_cat",
            ] = None
        elif case == "boolean_id":
            invalid["id"] = (
                invalid["id"] > 0
            )
        elif case == "nonnumeric_id":
            invalid["id"] = (
                invalid["id"]
                .astype(str)
            )
        elif case == "nonfinite_id":
            invalid["id"] = (
                invalid["id"]
                .astype(np.float64)
            )
            invalid.loc[0, "id"] = np.inf
        elif case == "fractional_id":
            invalid["id"] = (
                invalid["id"]
                .astype(np.float64)
            )
            invalid.loc[0, "id"] = 1.5
        elif case == "nonpositive_id":
            invalid.loc[0, "id"] = 0
        elif case == "duplicate_id":
            invalid.loc[
                1,
                "id",
            ] = invalid.loc[0, "id"]
        elif case == "boolean_label":
            invalid["label"] = (
                invalid["label"]
                .astype(bool)
            )
        elif case == "nonnumeric_label":
            invalid["label"] = (
                invalid["label"]
                .astype(str)
            )
        elif case == "nonfinite_label":
            invalid["label"] = (
                invalid["label"]
                .astype(np.float64)
            )
            invalid.loc[
                0,
                "label",
            ] = np.inf
        elif case == "invalid_label":
            invalid.loc[0, "label"] = 2
        elif case == "nonstr_category":
            invalid["attack_cat"] = (
                invalid["attack_cat"]
                .astype(object)
            )
            invalid.loc[
                0,
                "attack_cat",
            ] = 7
        elif case == "invalid_categories":
            invalid.loc[
                9,
                "attack_cat",
            ] = "Unknown"
        elif (
            case
            == "inconsistent_category_label"
        ):
            invalid.loc[0, "label"] = 1
        else:
            raise AssertionError(case)

    with pytest.raises(
        exception_type,
        match=message,
    ):
        align_unsw_evaluation_data(
            invalid,
            errors,
            predictions,
        )


def test_alignment_rejects_prediction_id_mismatch() -> None:
    raw_test, errors, predictions = (
        _evaluation_fixture()
    )

    identifiers = list(
        predictions.index
    )
    identifiers[0] = "unsw_testing:999"

    predictions.index = pd.Index(
        identifiers,
        name="flow_id",
    )

    with pytest.raises(
        ValueError,
        match="Prediction flow IDs",
    ):
        align_unsw_evaluation_data(
            raw_test,
            errors,
            predictions,
        )


def test_alignment_rejects_hidden_label_id_mismatch() -> None:
    raw_test, errors, predictions = (
        _evaluation_fixture()
    )

    raw_test.loc[0, "id"] = 999

    with pytest.raises(
        ValueError,
        match="Hidden-label flow IDs",
    ):
        align_unsw_evaluation_data(
            raw_test,
            errors,
            predictions,
        )


@pytest.mark.parametrize(
    ("case", "exception_type", "message"),
    (
        (
            "boolean_errors",
            TypeError,
            "errors must contain numeric",
        ),
        (
            "nonnumeric_errors",
            TypeError,
            "errors must contain numeric",
        ),
        (
            "nonfinite_errors",
            ValueError,
            "errors must contain finite",
        ),
        (
            "negative_errors",
            ValueError,
            "errors must be nonnegative",
        ),
        (
            "boolean_predictions",
            TypeError,
            "predictions must contain numeric",
        ),
        (
            "nonnumeric_predictions",
            TypeError,
            "predictions must contain numeric",
        ),
        (
            "nonfinite_predictions",
            ValueError,
            "predictions must contain finite",
        ),
        (
            "invalid_predictions",
            ValueError,
            "predictions must contain only",
        ),
    ),
)
def test_alignment_rejects_invalid_values(
    case: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    raw_test, errors, predictions = (
        _evaluation_fixture()
    )

    if case == "boolean_errors":
        errors = errors.astype(bool)
    elif case == "nonnumeric_errors":
        errors = errors.astype(object)
        errors.iloc[0] = "invalid"
    elif case == "nonfinite_errors":
        errors.iloc[0] = np.inf
    elif case == "negative_errors":
        errors.iloc[0] = -1.0
    elif case == "boolean_predictions":
        predictions = predictions.astype(
            bool
        )
    elif case == "nonnumeric_predictions":
        predictions = (
            predictions.astype(object)
        )
        predictions.iloc[0] = "invalid"
    elif case == "nonfinite_predictions":
        predictions = predictions.astype(
            np.float64
        )
        predictions.iloc[0] = np.inf
    elif case == "invalid_predictions":
        predictions.iloc[0] = 2
    else:
        raise AssertionError(case)

    with pytest.raises(
        exception_type,
        match=message,
    ):
        align_unsw_evaluation_data(
            raw_test,
            errors,
            predictions,
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
            "missing_scenario",
            ValueError,
            "scenario contains missing",
        ),
        (
            "nonstr_scenario",
            TypeError,
            "scenario values must be strings",
        ),
        (
            "invalid_scenarios",
            ValueError,
            "exactly the official",
        ),
        (
            "boolean_true",
            TypeError,
            "true_anomaly must contain numeric",
        ),
        (
            "nonnumeric_true",
            TypeError,
            "true_anomaly must contain numeric",
        ),
        (
            "nonfinite_true",
            ValueError,
            "true_anomaly must contain finite",
        ),
        (
            "invalid_true",
            ValueError,
            "true_anomaly must contain only",
        ),
        (
            "boolean_prediction",
            TypeError,
            "predicted_anomaly must contain numeric",
        ),
        (
            "nonnumeric_prediction",
            TypeError,
            "predicted_anomaly must contain numeric",
        ),
        (
            "nonfinite_prediction",
            ValueError,
            "predicted_anomaly must contain finite",
        ),
        (
            "invalid_prediction",
            ValueError,
            "predicted_anomaly must contain only",
        ),
        (
            "boolean_error",
            TypeError,
            "error must contain numeric",
        ),
        (
            "nonnumeric_error",
            TypeError,
            "error must contain numeric",
        ),
        (
            "nonfinite_error",
            ValueError,
            "error must contain finite",
        ),
        (
            "negative_error",
            ValueError,
            "error must be nonnegative",
        ),
        (
            "inconsistent_label",
            ValueError,
            "inconsistent labels",
        ),
    ),
)
def test_category_evaluation_rejects_invalid_data(
    case: str,
    exception_type: type[Exception],
    message: str,
) -> None:
    evaluation: object = (
        _aligned_evaluation()
    )

    if case == "not_dataframe":
        evaluation = object()
    elif case == "empty":
        evaluation = evaluation.iloc[
            0:0
        ].copy()
    elif case == "wrong_columns":
        evaluation = evaluation.rename(
            columns={
                "scenario": "wrong_scenario"
            }
        )
    elif case == "unnamed_index":
        evaluation.index = (
            evaluation.index.rename(None)
        )
    elif case == "missing_id":
        identifiers = list(
            evaluation.index
        )
        identifiers[0] = None
        evaluation.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "duplicate_id":
        identifiers = list(
            evaluation.index
        )
        identifiers[1] = identifiers[0]
        evaluation.index = pd.Index(
            identifiers,
            name="flow_id",
        )
    elif case == "missing_scenario":
        evaluation.loc[
            evaluation.index[0],
            "scenario",
        ] = None
    elif case == "nonstr_scenario":
        evaluation.loc[
            evaluation.index[0],
            "scenario",
        ] = 7
    elif case == "invalid_scenarios":
        evaluation.loc[
            evaluation.index[-1],
            "scenario",
        ] = "Unknown"
    elif case == "boolean_true":
        evaluation["true_anomaly"] = (
            evaluation["true_anomaly"]
            .astype(bool)
        )
    elif case == "nonnumeric_true":
        evaluation["true_anomaly"] = (
            evaluation["true_anomaly"]
            .astype(object)
        )
        evaluation.loc[
            evaluation.index[0],
            "true_anomaly",
        ] = "invalid"
    elif case == "nonfinite_true":
        evaluation["true_anomaly"] = (
            evaluation["true_anomaly"]
            .astype(np.float64)
        )
        evaluation.loc[
            evaluation.index[0],
            "true_anomaly",
        ] = np.inf
    elif case == "invalid_true":
        evaluation.loc[
            evaluation.index[0],
            "true_anomaly",
        ] = 2
    elif case == "boolean_prediction":
        evaluation[
            "predicted_anomaly"
        ] = evaluation[
            "predicted_anomaly"
        ].astype(bool)
    elif case == "nonnumeric_prediction":
        evaluation[
            "predicted_anomaly"
        ] = evaluation[
            "predicted_anomaly"
        ].astype(object)
        evaluation.loc[
            evaluation.index[0],
            "predicted_anomaly",
        ] = "invalid"
    elif case == "nonfinite_prediction":
        evaluation[
            "predicted_anomaly"
        ] = evaluation[
            "predicted_anomaly"
        ].astype(np.float64)
        evaluation.loc[
            evaluation.index[0],
            "predicted_anomaly",
        ] = np.inf
    elif case == "invalid_prediction":
        evaluation.loc[
            evaluation.index[0],
            "predicted_anomaly",
        ] = 2
    elif case == "boolean_error":
        evaluation[
            "reconstruction_error"
        ] = evaluation[
            "reconstruction_error"
        ].astype(bool)
    elif case == "nonnumeric_error":
        evaluation[
            "reconstruction_error"
        ] = evaluation[
            "reconstruction_error"
        ].astype(object)
        evaluation.loc[
            evaluation.index[0],
            "reconstruction_error",
        ] = "invalid"
    elif case == "nonfinite_error":
        evaluation.loc[
            evaluation.index[0],
            "reconstruction_error",
        ] = np.inf
    elif case == "negative_error":
        evaluation.loc[
            evaluation.index[0],
            "reconstruction_error",
        ] = -1.0
    elif case == "inconsistent_label":
        evaluation.loc[
            evaluation.index[0],
            "true_anomaly",
        ] = 1
    else:
        raise AssertionError(case)

    with pytest.raises(
        exception_type,
        match=message,
    ):
        evaluate_unsw_attack_categories(
            evaluation
        )
