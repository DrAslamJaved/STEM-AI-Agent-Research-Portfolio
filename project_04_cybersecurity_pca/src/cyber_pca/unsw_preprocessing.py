"""Leakage-safe preprocessing for UNSW-NB15."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from cyber_pca.unsw_data import (
    UNSWNB15Data,
    UNSW_CATEGORICAL_COLUMNS,
    UNSW_CURATED_COLUMNS,
)


DEFAULT_UNSW_NORMAL_FIT_FRACTION = 0.75
DEFAULT_UNSW_SPLIT_RANDOM_SEED = 42
DEFAULT_UNSW_PREPROCESSING_EVIDENCE_PATH = Path(
    "reports/validation/"
    "phase_07_unsw_nb15_preprocessing.json"
)

UNSW_EXCLUDED_MODEL_COLUMNS = (
    "id",
    "label",
    "attack_cat",
)
UNSW_MODEL_NUMERIC_COLUMNS = tuple(
    column
    for column in UNSW_CURATED_COLUMNS
    if column
    not in (
        *UNSW_EXCLUDED_MODEL_COLUMNS,
        *UNSW_CATEGORICAL_COLUMNS,
    )
)


@dataclass(frozen=True)
class UNSWRawDataSplits:
    """Raw normal-development and test partitions."""

    normal_fit: pd.DataFrame
    normal_calibration: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class UNSWPreprocessor:
    """Fitted normal-only encoder and scaler."""

    encoder: OneHotEncoder
    scaler: StandardScaler
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class UNSWStandardizedDataSplits:
    """Standardized model matrices and preprocessor."""

    normal_fit: pd.DataFrame
    normal_calibration: pd.DataFrame
    test: pd.DataFrame
    preprocessor: UNSWPreprocessor


def split_unsw_normal_calibration_test(
    data: UNSWNB15Data,
    normal_fit_fraction: float = (
        DEFAULT_UNSW_NORMAL_FIT_FRACTION
    ),
    random_seed: int = (
        DEFAULT_UNSW_SPLIT_RANDOM_SEED
    ),
) -> UNSWRawDataSplits:
    """Create deterministic normal-only development splits."""

    if not isinstance(data, UNSWNB15Data):
        raise TypeError(
            "data must be a UNSWNB15Data instance."
        )

    if (
        isinstance(normal_fit_fraction, bool)
        or not isinstance(
            normal_fit_fraction,
            Real,
        )
    ):
        raise TypeError(
            "normal_fit_fraction must be numeric."
        )

    fit_fraction = float(
        normal_fit_fraction
    )

    if (
        not 0.0 < fit_fraction < 1.0
    ):
        raise ValueError(
            "normal_fit_fraction must be "
            "strictly between 0 and 1."
        )

    if (
        isinstance(random_seed, bool)
        or not isinstance(
            random_seed,
            Integral,
        )
    ):
        raise TypeError(
            "random_seed must be an integer."
        )

    seed = int(random_seed)

    if seed < 0:
        raise ValueError(
            "random_seed must be nonnegative."
        )

    named_frames = (
        ("training", data.training),
        ("testing", data.testing),
    )

    for name, frame in named_frames:
        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas "
                "DataFrame."
            )

        if frame.empty:
            raise ValueError(
                f"{name} must not be empty."
            )

        if tuple(frame.columns) != (
            UNSW_CURATED_COLUMNS
        ):
            raise ValueError(
                f"{name} columns must exactly "
                "match UNSW_CURATED_COLUMNS."
            )

        if frame["id"].isna().any():
            raise ValueError(
                f"{name} contains missing IDs."
            )

        if frame["id"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate IDs."
            )

    normal_training = data.training.loc[
        data.training["label"].eq(0)
    ].copy(deep=True)

    if normal_training.shape[0] < 2:
        raise ValueError(
            "training must contain at least two "
            "normal observations."
        )

    normal_fit, normal_calibration = (
        train_test_split(
            normal_training,
            train_size=fit_fraction,
            random_state=seed,
            shuffle=True,
        )
    )

    normal_fit = (
        normal_fit.sort_values("id")
        .reset_index(drop=True)
    )
    normal_calibration = (
        normal_calibration.sort_values("id")
        .reset_index(drop=True)
    )

    test = data.testing.copy(deep=True)

    return UNSWRawDataSplits(
        normal_fit=normal_fit,
        normal_calibration=(
            normal_calibration
        ),
        test=test,
    )


def _validate_model_frame(
    frame: object,
    name: str,
) -> pd.DataFrame:
    """Validate model inputs without test-label access."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if frame.empty:
        raise ValueError(
            f"{name} must not be empty."
        )

    if tuple(frame.columns) != (
        UNSW_CURATED_COLUMNS
    ):
        raise ValueError(
            f"{name} columns must exactly match "
            "UNSW_CURATED_COLUMNS."
        )

    if frame["id"].isna().any():
        raise ValueError(
            f"{name} contains missing IDs."
        )

    if frame["id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate IDs."
        )

    model_columns = (
        *UNSW_MODEL_NUMERIC_COLUMNS,
        *UNSW_CATEGORICAL_COLUMNS,
    )

    if frame.loc[
        :,
        model_columns,
    ].isna().any().any():
        raise ValueError(
            f"{name} contains missing model "
            "values."
        )

    try:
        numeric_values = frame.loc[
            :,
            UNSW_MODEL_NUMERIC_COLUMNS,
        ].to_numpy(
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exception:
        raise TypeError(
            f"{name} numeric model columns "
            "must be numeric."
        ) from exception

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            f"{name} contains nonfinite model "
            "values."
        )

    return frame


def _combined_model_matrix(
    frame: pd.DataFrame,
    encoder: OneHotEncoder,
) -> np.ndarray:
    """Combine numeric and encoded inputs."""

    numeric_values = frame.loc[
        :,
        UNSW_MODEL_NUMERIC_COLUMNS,
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    encoded_values = encoder.transform(
        frame.loc[
            :,
            UNSW_CATEGORICAL_COLUMNS,
        ]
    )

    return np.column_stack(
        (
            numeric_values,
            encoded_values,
        )
    ).astype(
        np.float64,
        copy=False,
    )


def _flow_index(
    frame: pd.DataFrame,
    source_partition: str,
) -> pd.Index:
    """Create globally unique composite flow IDs."""

    identifiers = (
        source_partition
        + ":"
        + frame["id"].astype(str)
    )

    return pd.Index(
        identifiers.to_numpy(
            dtype=str,
        ),
        name="flow_id",
    )


def _standardized_frame(
    values: np.ndarray,
    frame: pd.DataFrame,
    source_partition: str,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Build one standardized float64 frame."""

    return pd.DataFrame(
        values,
        index=_flow_index(
            frame,
            source_partition,
        ),
        columns=feature_names,
        dtype=np.float64,
    )


def standardize_unsw_splits(
    splits: UNSWRawDataSplits,
) -> UNSWStandardizedDataSplits:
    """Encode and standardize with normal fitting data."""

    if not isinstance(
        splits,
        UNSWRawDataSplits,
    ):
        raise TypeError(
            "splits must be a "
            "UNSWRawDataSplits instance."
        )

    normal_fit = _validate_model_frame(
        splits.normal_fit,
        "normal_fit",
    )
    normal_calibration = (
        _validate_model_frame(
            splits.normal_calibration,
            "normal_calibration",
        )
    )
    test = _validate_model_frame(
        splits.test,
        "test",
    )

    if not normal_fit["label"].eq(0).all():
        raise ValueError(
            "normal_fit must contain only "
            "normal observations."
        )

    if not normal_calibration[
        "label"
    ].eq(0).all():
        raise ValueError(
            "normal_calibration must contain "
            "only normal observations."
        )

    fit_ids = set(normal_fit["id"])
    calibration_ids = set(
        normal_calibration["id"]
    )

    if not fit_ids.isdisjoint(
        calibration_ids
    ):
        raise ValueError(
            "Normal fitting and calibration "
            "IDs overlap."
        )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    encoder.fit(
        normal_fit.loc[
            :,
            UNSW_CATEGORICAL_COLUMNS,
        ]
    )

    encoded_feature_names = tuple(
        str(name)
        for name
        in encoder.get_feature_names_out(
            UNSW_CATEGORICAL_COLUMNS
        )
    )

    feature_names = (
        *UNSW_MODEL_NUMERIC_COLUMNS,
        *encoded_feature_names,
    )

    fit_matrix = _combined_model_matrix(
        normal_fit,
        encoder,
    )

    fit_variances = np.var(
        fit_matrix,
        axis=0,
        dtype=np.float64,
    )

    zero_variance_names = tuple(
        feature_names[index]
        for index in np.flatnonzero(
            fit_variances <= 0.0
        )
    )

    if zero_variance_names:
        raise ValueError(
            "normal_fit contains zero-variance "
            "model features: "
            f"{zero_variance_names}."
        )

    calibration_matrix = (
        _combined_model_matrix(
            normal_calibration,
            encoder,
        )
    )
    test_matrix = _combined_model_matrix(
        test,
        encoder,
    )

    scaler = StandardScaler()
    standardized_fit = scaler.fit_transform(
        fit_matrix
    )
    standardized_calibration = (
        scaler.transform(
            calibration_matrix
        )
    )
    standardized_test = scaler.transform(
        test_matrix
    )

    preprocessor = UNSWPreprocessor(
        encoder=encoder,
        scaler=scaler,
        feature_names=feature_names,
    )

    return UNSWStandardizedDataSplits(
        normal_fit=_standardized_frame(
            standardized_fit,
            normal_fit,
            "unsw_training",
            feature_names,
        ),
        normal_calibration=(
            _standardized_frame(
                standardized_calibration,
                normal_calibration,
                "unsw_training",
                feature_names,
            )
        ),
        test=_standardized_frame(
            standardized_test,
            test,
            "unsw_testing",
            feature_names,
        ),
        preprocessor=preprocessor,
    )

def _text_hash(
    values: list[str],
) -> str:
    """Hash an ordered sequence of text values."""

    return sha256(
        "\n".join(values).encode("utf-8")
    ).hexdigest()


def _float_array_hash(
    values: np.ndarray,
) -> str:
    """Hash float64 values with fixed byte order."""

    normalized = np.ascontiguousarray(
        values,
        dtype="<f8",
    )

    return sha256(
        normalized.tobytes()
    ).hexdigest()


def build_unsw_preprocessing_evidence(
    raw_splits: UNSWRawDataSplits,
    standardized: UNSWStandardizedDataSplits,
) -> dict[str, object]:
    """Build deterministic Phase 7 preprocessing evidence."""

    if not isinstance(
        raw_splits,
        UNSWRawDataSplits,
    ):
        raise TypeError(
            "raw_splits must be a "
            "UNSWRawDataSplits instance."
        )

    if not isinstance(
        standardized,
        UNSWStandardizedDataSplits,
    ):
        raise TypeError(
            "standardized must be a "
            "UNSWStandardizedDataSplits instance."
        )

    named_pairs = (
        (
            "normal_fit",
            raw_splits.normal_fit,
            standardized.normal_fit,
        ),
        (
            "normal_calibration",
            raw_splits.normal_calibration,
            standardized.normal_calibration,
        ),
        (
            "test",
            raw_splits.test,
            standardized.test,
        ),
    )

    feature_names = (
        standardized
        .preprocessor
        .feature_names
    )

    for name, raw_frame, model_frame in (
        named_pairs
    ):
        if not isinstance(
            raw_frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} raw partition must be "
                "a pandas DataFrame."
            )

        if not isinstance(
            model_frame,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} standardized partition "
                "must be a pandas DataFrame."
            )

        if raw_frame.shape[0] != (
            model_frame.shape[0]
        ):
            raise ValueError(
                f"{name} raw and standardized "
                "row counts differ."
            )

        if tuple(model_frame.columns) != (
            feature_names
        ):
            raise ValueError(
                f"{name} standardized columns "
                "do not match feature_names."
            )

    fit_values = (
        standardized.normal_fit.to_numpy(
            dtype=np.float64,
        )
    )
    calibration_values = (
        standardized
        .normal_calibration
        .to_numpy(
            dtype=np.float64,
        )
    )
    test_values = (
        standardized.test.to_numpy(
            dtype=np.float64,
        )
    )

    all_values_finite = bool(
        np.isfinite(fit_values).all()
        and np.isfinite(
            calibration_values
        ).all()
        and np.isfinite(test_values).all()
    )

    if not all_values_finite:
        raise ValueError(
            "standardized partitions must "
            "contain only finite values."
        )

    fitting_mean_error = float(
        np.max(
            np.abs(
                np.mean(
                    fit_values,
                    axis=0,
                    dtype=np.float64,
                )
            )
        )
    )
    fitting_std_error = float(
        np.max(
            np.abs(
                np.std(
                    fit_values,
                    axis=0,
                    ddof=0,
                    dtype=np.float64,
                )
                - 1.0
            )
        )
    )

    encoder = (
        standardized.preprocessor.encoder
    )
    scaler = (
        standardized.preprocessor.scaler
    )

    encoded_categorical_count = sum(
        len(categories)
        for categories in encoder.categories_
    )

    encoder_domain_values = [
        f"{column}={value}"
        for column, categories in zip(
            UNSW_CATEGORICAL_COLUMNS,
            encoder.categories_,
            strict=True,
        )
        for value in categories.astype(str)
    ]

    scaler_state = np.concatenate(
        (
            scaler.mean_,
            scaler.scale_,
        )
    )

    fit_ids = (
        raw_splits.normal_fit["id"]
        .astype(str)
        .tolist()
    )
    calibration_ids = (
        raw_splits
        .normal_calibration["id"]
        .astype(str)
        .tolist()
    )
    test_ids = (
        raw_splits.test["id"]
        .astype(str)
        .tolist()
    )

    training_attacks_excluded = bool(
        raw_splits.normal_fit[
            "label"
        ].eq(0).all()
        and raw_splits.normal_calibration[
            "label"
        ].eq(0).all()
    )

    scaler_sample_count = int(
        scaler.n_samples_seen_
    )

    return {
        "dataset": "UNSW-NB15",
        "phase": 7,
        "status": "passed",
        "partitions": {
            "normal_fit": {
                "observations": int(
                    raw_splits
                    .normal_fit
                    .shape[0]
                ),
                "id_sha256": _text_hash(
                    fit_ids
                ),
            },
            "normal_calibration": {
                "observations": int(
                    raw_splits
                    .normal_calibration
                    .shape[0]
                ),
                "id_sha256": _text_hash(
                    calibration_ids
                ),
            },
            "test": {
                "observations": int(
                    raw_splits.test.shape[0]
                ),
                "id_sha256": _text_hash(
                    test_ids
                ),
            },
        },
        "features": {
            "numeric": len(
                UNSW_MODEL_NUMERIC_COLUMNS
            ),
            "categorical_inputs": len(
                UNSW_CATEGORICAL_COLUMNS
            ),
            "encoded_categorical": int(
                encoded_categorical_count
            ),
            "model_features": len(
                feature_names
            ),
            "feature_name_sha256": (
                _text_hash(
                    list(feature_names)
                )
            ),
            "encoder_domain_sha256": (
                _text_hash(
                    encoder_domain_values
                )
            ),
        },
        "standardization": {
            "fit_sample_count": (
                scaler_sample_count
            ),
            "maximum_absolute_fitting_mean": (
                fitting_mean_error
            ),
            "maximum_fitting_std_error": (
                fitting_std_error
            ),
            "all_values_finite": (
                all_values_finite
            ),
            "scaler_state_sha256": (
                _float_array_hash(
                    scaler_state
                )
            ),
            "normal_fit_sha256": (
                _float_array_hash(
                    fit_values
                )
            ),
            "normal_calibration_sha256": (
                _float_array_hash(
                    calibration_values
                )
            ),
            "test_sha256": (
                _float_array_hash(
                    test_values
                )
            ),
        },
        "guards": {
            "training_attacks_excluded": (
                training_attacks_excluded
            ),
            "encoder_fit_normal_only": (
                scaler_sample_count
                == raw_splits
                .normal_fit
                .shape[0]
            ),
            "scaler_fit_normal_only": (
                scaler_sample_count
                == raw_splits
                .normal_fit
                .shape[0]
            ),
            "test_labels_used_for_fitting": False,
            "pca_fitted": False,
            "threshold_calibrated": False,
        },
    }


def write_unsw_preprocessing_evidence(
    raw_splits: UNSWRawDataSplits,
    standardized: UNSWStandardizedDataSplits,
    *,
    output_path: str | Path = (
        DEFAULT_UNSW_PREPROCESSING_EVIDENCE_PATH
    ),
) -> Path:
    """Write deterministic Phase 7 evidence JSON."""

    if not isinstance(
        output_path,
        (str, Path),
    ):
        raise TypeError(
            "output_path must be a string or Path."
        )

    evidence = (
        build_unsw_preprocessing_evidence(
            raw_splits,
            standardized,
        )
    )

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return destination
