"""Boundary validation for synthetic evaluation."""

import numpy as np
import pandas as pd
import pytest

from cyber_pca.evaluation import evaluate_scenarios


def _valid_evaluation_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_anomaly": [0, 1, 1, 1, 1],
            "predicted_anomaly": [0, 1, 1, 0, 1],
            "scenario": [
                "normal",
                "brute_force",
                "dos",
                "exfiltration",
                "port_scan",
            ],
            "reconstruction_error": [
                0.02,
                1.25,
                12.0,
                0.15,
                3.5,
            ],
        },
        index=pd.Index(
            [
                "flow-normal",
                "flow-brute",
                "flow-dos",
                "flow-exfiltration",
                "flow-port-scan",
            ],
            name="flow_id",
        ),
    )


@pytest.mark.parametrize(
    "invalid_data",
    [None, [], {}],
)
def test_scenarios_reject_non_dataframe(
    invalid_data: object,
) -> None:
    with pytest.raises(TypeError):
        evaluate_scenarios(invalid_data)  # type: ignore[arg-type]


def test_scenarios_reject_empty_dataframe() -> None:
    empty = _valid_evaluation_data().iloc[0:0]

    with pytest.raises(ValueError):
        evaluate_scenarios(empty)


def test_scenarios_reject_incorrect_columns() -> None:
    invalid = _valid_evaluation_data().drop(
        columns="reconstruction_error"
    )

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_incorrect_index_name() -> None:
    invalid = _valid_evaluation_data()
    invalid.index.name = "record_id"

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_missing_flow_identifiers() -> None:
    invalid = _valid_evaluation_data()
    identifiers = invalid.index.tolist()
    identifiers[0] = None
    invalid.index = pd.Index(
        identifiers,
        name="flow_id",
    )

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_duplicate_flow_identifiers() -> None:
    invalid = _valid_evaluation_data()
    identifiers = invalid.index.tolist()
    identifiers[-1] = identifiers[0]
    invalid.index = pd.Index(
        identifiers,
        name="flow_id",
    )

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_missing_scenario_values() -> None:
    invalid = _valid_evaluation_data()
    invalid.loc["flow-normal", "scenario"] = None

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_nonstring_scenario_values() -> None:
    invalid = _valid_evaluation_data()
    invalid.loc["flow-normal", "scenario"] = 7

    with pytest.raises(TypeError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_unknown_scenarios() -> None:
    invalid = _valid_evaluation_data()
    invalid.loc[
        "flow-port-scan",
        "scenario",
    ] = "lateral_movement"

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_require_every_contract_scenario() -> None:
    invalid = _valid_evaluation_data().drop(
        index="flow-port-scan"
    )

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_inconsistent_true_labels() -> None:
    invalid = _valid_evaluation_data()
    invalid.loc[
        "flow-brute",
        "true_anomaly",
    ] = 0

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_boolean_predictions() -> None:
    invalid = _valid_evaluation_data()
    invalid["predicted_anomaly"] = (
        invalid["predicted_anomaly"].astype(bool)
    )

    with pytest.raises(TypeError):
        evaluate_scenarios(invalid)


def test_scenarios_reject_nonnumeric_true_labels() -> None:
    invalid = _valid_evaluation_data()
    invalid["true_anomaly"] = (
        invalid["true_anomaly"].astype(object)
    )
    invalid.loc[
        "flow-normal",
        "true_anomaly",
    ] = "normal"

    with pytest.raises(TypeError):
        evaluate_scenarios(invalid)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf],
)
def test_scenarios_reject_nonfinite_predictions(
    invalid_value: float,
) -> None:
    invalid = _valid_evaluation_data()
    invalid["predicted_anomaly"] = (
        invalid["predicted_anomaly"].astype(
            np.float64
        )
    )
    invalid.loc[
        "flow-normal",
        "predicted_anomaly",
    ] = invalid_value

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


@pytest.mark.parametrize(
    "invalid_value",
    [-1.0, 0.5, 2.0],
)
def test_scenarios_reject_nonbinary_predictions(
    invalid_value: float,
) -> None:
    invalid = _valid_evaluation_data()
    invalid["predicted_anomaly"] = (
        invalid["predicted_anomaly"].astype(
            np.float64
        )
    )
    invalid.loc[
        "flow-normal",
        "predicted_anomaly",
    ] = invalid_value

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)


@pytest.mark.parametrize(
    "invalid_value",
    ["large", True],
)
def test_scenarios_reject_nonnumeric_errors(
    invalid_value: object,
) -> None:
    invalid = _valid_evaluation_data()
    invalid["reconstruction_error"] = (
        invalid[
            "reconstruction_error"
        ].astype(object)
    )
    invalid.loc[
        "flow-normal",
        "reconstruction_error",
    ] = invalid_value

    with pytest.raises(TypeError):
        evaluate_scenarios(invalid)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf, -0.1],
)
def test_scenarios_reject_invalid_errors(
    invalid_value: float,
) -> None:
    invalid = _valid_evaluation_data()
    invalid.loc[
        "flow-normal",
        "reconstruction_error",
    ] = invalid_value

    with pytest.raises(ValueError):
        evaluate_scenarios(invalid)
