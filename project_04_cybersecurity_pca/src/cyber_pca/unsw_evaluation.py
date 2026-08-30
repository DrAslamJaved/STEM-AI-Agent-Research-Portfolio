"""Evaluation of frozen predictions on official UNSW-NB15 labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


UNSW_ATTACK_CATEGORY_ORDER = (
    "Normal",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
)

UNSW_ATTACK_CATEGORY_LABELS = {
    category: (
        0
        if category == "Normal"
        else 1
    )
    for category in (
        UNSW_ATTACK_CATEGORY_ORDER
    )
}


def _validate_indexed_series(
    series: object,
    *,
    name: str,
    expected_name: str,
) -> pd.Series:
    """Validate one frozen indexed detector series."""

    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name} must be a pandas Series."
        )

    if series.empty:
        raise ValueError(
            f"{name} must not be empty."
        )

    if series.name != expected_name:
        raise ValueError(
            f"{name} must be named "
            f"{expected_name!r}."
        )

    if series.index.name != "flow_id":
        raise ValueError(
            f"{name} index must be named "
            "flow_id."
        )

    if series.index.hasnans:
        raise ValueError(
            f"{name} contains missing flow IDs."
        )

    if series.index.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate flow IDs."
        )

    if series.isna().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    return series


def align_unsw_evaluation_data(
    raw_test: pd.DataFrame,
    reconstruction_errors: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Align hidden test labels to frozen outputs by partition ID."""

    if not isinstance(
        raw_test,
        pd.DataFrame,
    ):
        raise TypeError(
            "raw_test must be a pandas "
            "DataFrame."
        )

    if raw_test.empty:
        raise ValueError(
            "raw_test must not be empty."
        )

    required_columns = {
        "id",
        "label",
        "attack_cat",
    }

    missing_columns = (
        required_columns
        - set(raw_test.columns)
    )

    if missing_columns:
        raise ValueError(
            "raw_test is missing required "
            f"columns: {sorted(missing_columns)}."
        )

    errors = _validate_indexed_series(
        reconstruction_errors,
        name="reconstruction_errors",
        expected_name=(
            "reconstruction_error"
        ),
    )

    frozen_predictions = (
        _validate_indexed_series(
            predictions,
            name="predictions",
            expected_name=(
                "predicted_anomaly"
            ),
        )
    )

    if (
        raw_test["id"].isna().any()
        or raw_test["label"].isna().any()
        or raw_test[
            "attack_cat"
        ].isna().any()
    ):
        raise ValueError(
            "raw_test evaluation columns "
            "contain missing values."
        )

    if (
        pd.api.types.is_bool_dtype(
            raw_test["id"].dtype
        )
        or not pd.api.types.is_numeric_dtype(
            raw_test["id"].dtype
        )
    ):
        raise TypeError(
            "raw_test IDs must be numeric."
        )

    id_values = raw_test["id"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(np.isfinite(id_values)):
        raise ValueError(
            "raw_test IDs must be finite."
        )

    if not np.all(
        id_values == np.floor(id_values)
    ):
        raise ValueError(
            "raw_test IDs must be integers."
        )

    integer_ids = id_values.astype(
        np.int64,
        copy=False,
    )

    if np.any(integer_ids <= 0):
        raise ValueError(
            "raw_test IDs must be positive."
        )

    if (
        pd.Index(integer_ids)
        .duplicated()
        .any()
    ):
        raise ValueError(
            "raw_test contains duplicate IDs."
        )

    label_series = raw_test["label"]

    if (
        pd.api.types.is_bool_dtype(
            label_series.dtype
        )
        or not pd.api.types.is_numeric_dtype(
            label_series.dtype
        )
    ):
        raise TypeError(
            "raw_test labels must be numeric."
        )

    label_values = label_series.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(
        np.isfinite(label_values)
    ):
        raise ValueError(
            "raw_test labels must be finite."
        )

    if not np.all(
        np.isin(
            label_values,
            (0.0, 1.0),
        )
    ):
        raise ValueError(
            "raw_test labels must contain "
            "only 0 and 1."
        )

    categories = raw_test[
        "attack_cat"
    ]

    if not all(
        isinstance(value, str)
        for value in categories
    ):
        raise TypeError(
            "raw_test attack categories "
            "must be strings."
        )

    category_values = categories.to_numpy(
        dtype=object,
        copy=True,
    )

    observed_categories = set(
        category_values.tolist()
    )

    if observed_categories != set(
        UNSW_ATTACK_CATEGORY_ORDER
    ):
        raise ValueError(
            "raw_test attack categories must "
            "contain exactly the official "
            "UNSW-NB15 categories."
        )

    expected_labels = np.asarray(
        [
            UNSW_ATTACK_CATEGORY_LABELS[
                str(category)
            ]
            for category in category_values
        ],
        dtype=np.int8,
    )

    integer_labels = label_values.astype(
        np.int8,
        copy=False,
    )

    if not np.array_equal(
        integer_labels,
        expected_labels,
    ):
        raise ValueError(
            "raw_test labels and attack "
            "categories are inconsistent."
        )

    raw_flow_ids = pd.Index(
        [
            f"unsw_testing:{value}"
            for value in integer_ids
        ],
        name="flow_id",
    )

    hidden_labels = pd.DataFrame(
        {
            "true_anomaly": integer_labels,
            "scenario": category_values,
        },
        index=raw_flow_ids,
    )

    error_ids = set(
        errors.index.tolist()
    )
    prediction_ids = set(
        frozen_predictions.index.tolist()
    )
    label_ids = set(
        hidden_labels.index.tolist()
    )

    if error_ids != prediction_ids:
        raise ValueError(
            "Prediction flow IDs do not "
            "match reconstruction-error IDs."
        )

    if error_ids != label_ids:
        raise ValueError(
            "Hidden-label flow IDs do not "
            "match frozen detector IDs."
        )

    ordered_index = errors.index.copy()

    aligned_predictions = (
        frozen_predictions.reindex(
            ordered_index
        )
    )

    aligned_labels = hidden_labels.reindex(
        ordered_index
    )

    if (
        pd.api.types.is_bool_dtype(
            errors.dtype
        )
        or not pd.api.types.is_numeric_dtype(
            errors.dtype
        )
    ):
        raise TypeError(
            "reconstruction_errors must "
            "contain numeric values."
        )

    error_values = errors.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.all(np.isfinite(error_values)):
        raise ValueError(
            "reconstruction_errors must "
            "contain finite values."
        )

    if np.any(error_values < 0.0):
        raise ValueError(
            "reconstruction_errors must be "
            "nonnegative."
        )

    if (
        pd.api.types.is_bool_dtype(
            aligned_predictions.dtype
        )
        or not pd.api.types.is_numeric_dtype(
            aligned_predictions.dtype
        )
    ):
        raise TypeError(
            "predictions must contain "
            "numeric binary values."
        )

    prediction_values = (
        aligned_predictions.to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )

    if not np.all(
        np.isfinite(prediction_values)
    ):
        raise ValueError(
            "predictions must contain "
            "finite values."
        )

    if not np.all(
        np.isin(
            prediction_values,
            (0.0, 1.0),
        )
    ):
        raise ValueError(
            "predictions must contain only "
            "0 and 1."
        )

    return pd.DataFrame(
        {
            "true_anomaly": (
                aligned_labels[
                    "true_anomaly"
                ].to_numpy(
                    dtype=np.int8,
                    copy=True,
                )
            ),
            "predicted_anomaly": (
                prediction_values.astype(
                    np.int8,
                    copy=False,
                )
            ),
            "scenario": (
                aligned_labels[
                    "scenario"
                ].to_numpy(
                    dtype=object,
                    copy=True,
                )
            ),
            "reconstruction_error": (
                error_values
            ),
        },
        index=ordered_index,
    )


def evaluate_unsw_attack_categories(
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize frozen predictions by official attack category."""

    if not isinstance(
        evaluation_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "evaluation_data must be a "
            "pandas DataFrame."
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

    if tuple(
        evaluation_data.columns
    ) != expected_columns:
        raise ValueError(
            "evaluation_data columns must "
            f"exactly match {expected_columns}."
        )

    if evaluation_data.index.name != "flow_id":
        raise ValueError(
            "evaluation_data index must be "
            "named flow_id."
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

    scenario_series = evaluation_data[
        "scenario"
    ]

    if scenario_series.isna().any():
        raise ValueError(
            "scenario contains missing values."
        )

    if not all(
        isinstance(value, str)
        for value in scenario_series
    ):
        raise TypeError(
            "scenario values must be strings."
        )

    if set(scenario_series) != set(
        UNSW_ATTACK_CATEGORY_ORDER
    ):
        raise ValueError(
            "scenario values must contain "
            "exactly the official UNSW-NB15 "
            "categories."
        )

    validated_binary: dict[
        str,
        np.ndarray,
    ] = {}

    for column in (
        "true_anomaly",
        "predicted_anomaly",
    ):
        series = evaluation_data[column]

        if (
            pd.api.types.is_bool_dtype(
                series.dtype
            )
            or not pd.api.types.is_numeric_dtype(
                series.dtype
            )
        ):
            raise TypeError(
                f"{column} must contain "
                "numeric binary values."
            )

        values = series.to_numpy(
            dtype=np.float64,
            copy=True,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{column} must contain "
                "finite values."
            )

        if not np.all(
            np.isin(
                values,
                (0.0, 1.0),
            )
        ):
            raise ValueError(
                f"{column} must contain only "
                "0 and 1."
            )

        validated_binary[column] = (
            values.astype(
                np.int8,
                copy=False,
            )
        )

    error_series = evaluation_data[
        "reconstruction_error"
    ]

    if (
        pd.api.types.is_bool_dtype(
            error_series.dtype
        )
        or not pd.api.types.is_numeric_dtype(
            error_series.dtype
        )
    ):
        raise TypeError(
            "reconstruction_error must "
            "contain numeric values."
        )

    error_values = (
        error_series.to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )

    if not np.all(np.isfinite(error_values)):
        raise ValueError(
            "reconstruction_error must "
            "contain finite values."
        )

    if np.any(error_values < 0.0):
        raise ValueError(
            "reconstruction_error must be "
            "nonnegative."
        )

    scenario_values = (
        scenario_series.to_numpy(
            dtype=object,
            copy=True,
        )
    )

    rows: list[
        dict[str, object]
    ] = []

    for category in (
        UNSW_ATTACK_CATEGORY_ORDER
    ):
        mask = (
            scenario_values == category
        )

        expected_label = (
            UNSW_ATTACK_CATEGORY_LABELS[
                category
            ]
        )

        true_values = validated_binary[
            "true_anomaly"
        ][mask]

        if not np.all(
            true_values == expected_label
        ):
            raise ValueError(
                f"Attack category {category!r} "
                "has inconsistent labels."
            )

        category_predictions = (
            validated_binary[
                "predicted_anomaly"
            ][mask]
        )

        category_errors = error_values[
            mask
        ]

        observations = int(mask.sum())

        predicted_anomaly = int(
            np.count_nonzero(
                category_predictions == 1
            )
        )

        predicted_normal = (
            observations
            - predicted_anomaly
        )

        rows.append(
            {
                "attack_category": category,
                "true_label": expected_label,
                "observations": observations,
                "predicted_normal": (
                    predicted_normal
                ),
                "predicted_anomaly": (
                    predicted_anomaly
                ),
                "predicted_anomaly_rate": (
                    predicted_anomaly
                    / observations
                ),
                "mean_reconstruction_error": float(
                    np.mean(
                        category_errors,
                        dtype=np.float64,
                    )
                ),
                "median_reconstruction_error": float(
                    np.median(
                        category_errors
                    )
                ),
                "maximum_reconstruction_error": float(
                    np.max(
                        category_errors
                    )
                ),
            }
        )

    result = pd.DataFrame.from_records(
        rows,
        columns=[
            "attack_category",
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

    for column in (
        "true_label",
        "observations",
        "predicted_normal",
        "predicted_anomaly",
    ):
        result[column] = result[
            column
        ].astype(np.int64)

    return result
