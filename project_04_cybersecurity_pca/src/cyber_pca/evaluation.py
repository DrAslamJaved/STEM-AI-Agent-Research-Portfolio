"""Label-based evaluation for frozen anomaly predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinaryEvaluationResult:
    """Immutable binary anomaly-evaluation metrics."""

    total: int
    normal_support: int
    anomaly_support: int
    predicted_normal: int
    predicted_anomaly: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: tuple[
        tuple[int, int],
        tuple[int, int],
    ]


def align_evaluation_data(
    raw_test: pd.DataFrame,
    reconstruction_errors: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Align labels, scenarios, errors, and predictions by flow ID."""

    if not isinstance(raw_test, pd.DataFrame):
        raise TypeError(
            "raw_test must be a pandas DataFrame."
        )

    if raw_test.empty:
        raise ValueError(
            "raw_test must not be empty."
        )

    if raw_test.columns.duplicated().any():
        raise ValueError(
            "raw_test contains duplicate columns."
        )

    required_columns = {
        "flow_id",
        "is_anomaly",
        "scenario",
    }

    missing_columns = (
        required_columns - set(raw_test.columns)
    )

    if missing_columns:
        raise ValueError(
            "raw_test is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    label_frame = raw_test.loc[
        :,
        [
            "flow_id",
            "is_anomaly",
            "scenario",
        ],
    ].copy(deep=True)

    if label_frame["flow_id"].isna().any():
        raise ValueError(
            "raw_test contains missing flow IDs."
        )

    if label_frame["flow_id"].duplicated().any():
        raise ValueError(
            "raw_test contains duplicate flow IDs."
        )

    if label_frame["is_anomaly"].isna().any():
        raise ValueError(
            "raw_test contains missing anomaly labels."
        )

    if label_frame["scenario"].isna().any():
        raise ValueError(
            "raw_test contains missing scenarios."
        )

    if not pd.api.types.is_numeric_dtype(
        label_frame["is_anomaly"].dtype
    ):
        raise TypeError(
            "raw_test anomaly labels must be numeric."
        )

    label_values = label_frame[
        "is_anomaly"
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(np.isfinite(label_values)):
        raise ValueError(
            "raw_test anomaly labels must be finite."
        )

    if not set(label_values).issubset({0.0, 1.0}):
        raise ValueError(
            "raw_test anomaly labels must be binary."
        )

    indexed_series = (
        (
            "reconstruction_errors",
            reconstruction_errors,
            "reconstruction_error",
        ),
        (
            "predictions",
            predictions,
            "is_anomaly",
        ),
    )

    for (
        argument_name,
        series,
        expected_name,
    ) in indexed_series:
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"{argument_name} must be a "
                "pandas Series."
            )

        if series.empty:
            raise ValueError(
                f"{argument_name} must not be empty."
            )

        if series.name != expected_name:
            raise ValueError(
                f"{argument_name} must be named "
                f"'{expected_name}'."
            )

        if series.index.name != "flow_id":
            raise ValueError(
                f"{argument_name} index must be "
                "named 'flow_id'."
            )

        if series.index.hasnans:
            raise ValueError(
                f"{argument_name} contains missing "
                "flow IDs."
            )

        if series.index.duplicated().any():
            raise ValueError(
                f"{argument_name} contains duplicate "
                "flow IDs."
            )

        if series.isna().any():
            raise ValueError(
                f"{argument_name} contains missing "
                "values."
            )

        if not pd.api.types.is_numeric_dtype(
            series.dtype
        ):
            raise TypeError(
                f"{argument_name} must contain "
                "numeric values."
            )

    error_values = reconstruction_errors.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    prediction_values = predictions.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(np.isfinite(error_values)):
        raise ValueError(
            "reconstruction_errors must be finite."
        )

    if np.any(error_values < 0.0):
        raise ValueError(
            "reconstruction_errors must be "
            "nonnegative."
        )

    if not np.all(np.isfinite(prediction_values)):
        raise ValueError(
            "predictions must be finite."
        )

    if not set(prediction_values).issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            "predictions must be binary."
        )

    raw_identifiers = set(
        label_frame["flow_id"].tolist()
    )
    error_identifiers = set(
        reconstruction_errors.index.tolist()
    )
    prediction_identifiers = set(
        predictions.index.tolist()
    )

    if raw_identifiers != error_identifiers:
        raise ValueError(
            "raw_test and reconstruction_errors "
            "flow IDs do not match."
        )

    if raw_identifiers != prediction_identifiers:
        raise ValueError(
            "raw_test and predictions flow IDs "
            "do not match."
        )

    ordered_index = predictions.index.copy()

    indexed_labels = label_frame.set_index(
        "flow_id"
    )

    aligned = pd.DataFrame(
        {
            "true_anomaly": (
                indexed_labels.loc[
                    ordered_index,
                    "is_anomaly",
                ]
                .to_numpy(
                    dtype=np.int8,
                    copy=True,
                )
            ),
            "predicted_anomaly": (
                prediction_values.astype(
                    np.int8,
                    copy=True,
                )
            ),
            "scenario": (
                indexed_labels.loc[
                    ordered_index,
                    "scenario",
                ]
                .to_numpy(copy=True)
            ),
            "reconstruction_error": (
                reconstruction_errors.loc[
                    ordered_index
                ]
                .to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
            ),
        },
        index=ordered_index,
    )

    aligned.index.name = "flow_id"

    return aligned

def evaluate_binary_predictions(
    evaluation_data: pd.DataFrame,
    *,
    zero_division: int = 0,
) -> BinaryEvaluationResult:
    """Calculate binary anomaly-detection metrics."""

    if not isinstance(evaluation_data, pd.DataFrame):
        raise TypeError(
            "evaluation_data must be a pandas "
            "DataFrame."
        )

    if evaluation_data.empty:
        raise ValueError(
            "evaluation_data must not be empty."
        )

    if evaluation_data.columns.duplicated().any():
        raise ValueError(
            "evaluation_data contains duplicate "
            "columns."
        )

    required_columns = {
        "true_anomaly",
        "predicted_anomaly",
    }

    missing_columns = (
        required_columns
        - set(evaluation_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "evaluation_data is missing required "
            f"columns: {sorted(missing_columns)}."
        )

    if evaluation_data.index.name != "flow_id":
        raise ValueError(
            "evaluation_data index must be named "
            "'flow_id'."
        )

    if evaluation_data.index.hasnans:
        raise ValueError(
            "evaluation_data contains missing "
            "flow IDs."
        )

    if evaluation_data.index.duplicated().any():
        raise ValueError(
            "evaluation_data contains duplicate "
            "flow IDs."
        )

    if (
        isinstance(zero_division, bool)
        or not isinstance(
            zero_division,
            (int, np.integer),
        )
    ):
        raise TypeError(
            "zero_division must be integer 0 or 1."
        )

    zero_division_value = int(zero_division)

    if zero_division_value not in {0, 1}:
        raise ValueError(
            "zero_division must be 0 or 1."
        )

    validated_values: dict[str, np.ndarray] = {}

    for column_name in (
        "true_anomaly",
        "predicted_anomaly",
    ):
        series = evaluation_data[column_name]

        if series.isna().any():
            raise ValueError(
                f"{column_name} contains missing "
                "values."
            )

        if (
            not pd.api.types.is_numeric_dtype(
                series.dtype
            )
            or pd.api.types.is_bool_dtype(
                series.dtype
            )
        ):
            raise TypeError(
                f"{column_name} must contain "
                "numeric binary values."
            )

        values = series.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{column_name} must contain "
                "finite values."
            )

        if not set(values).issubset({0.0, 1.0}):
            raise ValueError(
                f"{column_name} must contain only "
                "0 and 1."
            )

        validated_values[column_name] = (
            values.astype(
                np.int8,
                copy=False,
            )
        )

    y_true = validated_values["true_anomaly"]
    y_pred = validated_values[
        "predicted_anomaly"
    ]

    true_negatives = int(
        np.count_nonzero(
            (y_true == 0) & (y_pred == 0)
        )
    )
    false_positives = int(
        np.count_nonzero(
            (y_true == 0) & (y_pred == 1)
        )
    )
    false_negatives = int(
        np.count_nonzero(
            (y_true == 1) & (y_pred == 0)
        )
    )
    true_positives = int(
        np.count_nonzero(
            (y_true == 1) & (y_pred == 1)
        )
    )

    total = int(y_true.size)
    normal_support = (
        true_negatives + false_positives
    )
    anomaly_support = (
        false_negatives + true_positives
    )
    predicted_normal = (
        true_negatives + false_negatives
    )
    predicted_anomaly = (
        false_positives + true_positives
    )

    def safe_divide(
        numerator: int | float,
        denominator: int | float,
    ) -> float:
        if denominator == 0:
            return float(zero_division_value)

        return float(numerator / denominator)

    precision = safe_divide(
        true_positives,
        predicted_anomaly,
    )
    recall = safe_divide(
        true_positives,
        anomaly_support,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        true_negatives + true_positives,
        total,
    )

    false_positive_rate = safe_divide(
        false_positives,
        normal_support,
    )

    false_negative_rate = safe_divide(
        false_negatives,
        anomaly_support,
    )

    return BinaryEvaluationResult(
        total=total,
        normal_support=normal_support,
        anomaly_support=anomaly_support,
        predicted_normal=predicted_normal,
        predicted_anomaly=predicted_anomaly,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        confusion_matrix=(
            (
                true_negatives,
                false_positives,
            ),
            (
                false_negatives,
                true_positives,
            ),
        ),
    )

def evaluate_scenarios(
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize predictions and errors by traffic scenario."""

    if not isinstance(evaluation_data, pd.DataFrame):
        raise TypeError(
            "evaluation_data must be a pandas DataFrame."
        )

    if evaluation_data.empty:
        raise ValueError(
            "evaluation_data must not be empty."
        )

    expected_columns = (
        "true_anomaly",
        "predicted_anomaly",
        "scenario",
        "reconstruction_error",
    )

    if tuple(evaluation_data.columns) != expected_columns:
        raise ValueError(
            "evaluation_data columns must exactly match "
            f"{expected_columns}."
        )

    if evaluation_data.index.name != "flow_id":
        raise ValueError(
            "evaluation_data index must be named flow_id."
        )

    if evaluation_data.index.hasnans:
        raise ValueError(
            "evaluation_data contains missing flow IDs."
        )

    if evaluation_data.index.duplicated().any():
        raise ValueError(
            "evaluation_data contains duplicate flow IDs."
        )

    scenario_labels = {
        "normal": 0,
        "brute_force": 1,
        "dos": 1,
        "exfiltration": 1,
        "port_scan": 1,
    }

    scenario_order = tuple(scenario_labels)

    scenarios = evaluation_data["scenario"]

    if scenarios.isna().any():
        raise ValueError(
            "scenario contains missing values."
        )

    if not all(
        isinstance(value, str)
        for value in scenarios
    ):
        raise TypeError(
            "scenario values must be strings."
        )

    observed_scenarios = set(scenarios)

    if observed_scenarios != set(scenario_order):
        raise ValueError(
            "scenario values must contain exactly "
            f"{scenario_order}."
        )

    binary_values: dict[str, np.ndarray] = {}

    for column in (
        "true_anomaly",
        "predicted_anomaly",
    ):
        series = evaluation_data[column]

        if (
            pd.api.types.is_bool_dtype(series.dtype)
            or not pd.api.types.is_numeric_dtype(
                series.dtype
            )
        ):
            raise TypeError(
                f"{column} must contain numeric values."
            )

        values = series.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{column} contains nonfinite values."
            )

        if not np.all(np.isin(values, (0.0, 1.0))):
            raise ValueError(
                f"{column} must contain only 0 and 1."
            )

        binary_values[column] = values

    error_series = evaluation_data[
        "reconstruction_error"
    ]

    if (
        pd.api.types.is_bool_dtype(error_series.dtype)
        or not pd.api.types.is_numeric_dtype(
            error_series.dtype
        )
    ):
        raise TypeError(
            "reconstruction_error must contain "
            "numeric values."
        )

    error_values = error_series.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(np.isfinite(error_values)):
        raise ValueError(
            "reconstruction_error contains "
            "nonfinite values."
        )

    if np.any(error_values < 0.0):
        raise ValueError(
            "reconstruction_error must be nonnegative."
        )

    rows: list[dict[str, object]] = []

    for scenario in scenario_order:
        scenario_mask = (
            scenarios.to_numpy(copy=True) == scenario
        )

        expected_label = scenario_labels[scenario]
        true_labels = binary_values[
            "true_anomaly"
        ][scenario_mask]

        if not np.all(true_labels == expected_label):
            raise ValueError(
                f"Scenario {scenario!r} has labels "
                "inconsistent with its contract."
            )

        predictions = binary_values[
            "predicted_anomaly"
        ][scenario_mask]

        scenario_errors = error_values[
            scenario_mask
        ]

        observations = int(
            scenario_mask.sum()
        )

        predicted_anomaly = int(
            np.count_nonzero(predictions == 1.0)
        )

        predicted_normal = (
            observations - predicted_anomaly
        )

        rows.append(
            {
                "scenario": scenario,
                "true_label": expected_label,
                "observations": observations,
                "predicted_normal": predicted_normal,
                "predicted_anomaly": predicted_anomaly,
                "predicted_anomaly_rate": (
                    predicted_anomaly / observations
                ),
                "mean_reconstruction_error": float(
                    np.mean(
                        scenario_errors,
                        dtype=np.float64,
                    )
                ),
                "median_reconstruction_error": float(
                    np.median(scenario_errors)
                ),
                "maximum_reconstruction_error": float(
                    np.max(scenario_errors)
                ),
            }
        )

    result = pd.DataFrame.from_records(
        rows,
        columns=[
            "scenario",
            "true_label",
            "observations",
            "predicted_normal",
            "predicted_anomaly",
            "predicted_anomaly_rate",
            "mean_reconstruction_error",
            "median_reconstruction_error",
            "maximum_reconstruction_error",
        ],
    )

    integer_columns = (
        "true_label",
        "observations",
        "predicted_normal",
        "predicted_anomaly",
    )

    result.loc[:, integer_columns] = result.loc[
        :, integer_columns
    ].astype(np.int64)

    return result
