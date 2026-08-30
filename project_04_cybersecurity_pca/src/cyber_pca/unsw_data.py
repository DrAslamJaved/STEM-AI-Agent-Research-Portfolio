"""UNSW-NB15 raw-data acquisition and validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_UNSW_RAW_DIRECTORY = Path(
    "data/raw"
)
DEFAULT_UNSW_MANIFEST_PATH = Path(
    "reports/validation/"
    "phase_07_unsw_nb15_manifest.json"
)
UNSW_TRAINING_FILENAME = (
    "UNSW_NB15_training-set.csv"
)
UNSW_TESTING_FILENAME = (
    "UNSW_NB15_testing-set.csv"
)
UNSW_FEATURE_DESCRIPTIONS_FILENAME = (
    "NUSW-NB15_features.csv"
)
UNSW_CURATED_FILE_ENCODING = "utf-8"
UNSW_FEATURE_DESCRIPTION_ENCODING = "cp1252"
UNSW_EXPECTED_TRAINING_ROWS = 175341
UNSW_EXPECTED_TESTING_ROWS = 82332
UNSW_EXPECTED_FEATURE_DESCRIPTION_ROWS = 49

UNSW_CURATED_COLUMNS = (
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
)

UNSW_FEATURE_DESCRIPTION_COLUMNS = (
    "No.",
    "Name",
    "Type ",
    "Description",
)

UNSW_CATEGORICAL_COLUMNS = (
    "proto",
    "service",
    "state",
)


@dataclass(frozen=True)
class UNSWNB15Paths:
    """Resolved paths for the official raw files."""

    training: Path
    testing: Path
    feature_descriptions: Path


@dataclass(frozen=True)
class UNSWNB15Data:
    """Loaded official curated data and descriptions."""

    training: pd.DataFrame
    testing: pd.DataFrame
    feature_descriptions: pd.DataFrame


def resolve_unsw_nb15_paths(
    raw_directory: str | Path = (
        DEFAULT_UNSW_RAW_DIRECTORY
    ),
) -> UNSWNB15Paths:
    """Resolve official raw paths without reading files."""

    if not isinstance(
        raw_directory,
        (str, Path),
    ):
        raise TypeError(
            "raw_directory must be a string or Path."
        )

    root = Path(raw_directory)

    return UNSWNB15Paths(
        training=(
            root
            / UNSW_TRAINING_FILENAME
        ),
        testing=(
            root
            / UNSW_TESTING_FILENAME
        ),
        feature_descriptions=(
            root
            / UNSW_FEATURE_DESCRIPTIONS_FILENAME
        ),
    )

def load_unsw_nb15(
    source: UNSWNB15Paths | str | Path = (
        DEFAULT_UNSW_RAW_DIRECTORY
    ),
) -> UNSWNB15Data:
    """Load official files without modifying raw data."""

    if isinstance(source, UNSWNB15Paths):
        paths = source
    else:
        paths = resolve_unsw_nb15_paths(
            source
        )

    named_paths = (
        ("training", paths.training),
        ("testing", paths.testing),
        (
            "feature_descriptions",
            paths.feature_descriptions,
        ),
    )

    for name, file_path in named_paths:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{name} file does not exist: "
                f"{file_path}"
            )

    training = pd.read_csv(
        paths.training,
        encoding=UNSW_CURATED_FILE_ENCODING,
        low_memory=False,
    )
    testing = pd.read_csv(
        paths.testing,
        encoding=UNSW_CURATED_FILE_ENCODING,
        low_memory=False,
    )
    feature_descriptions = pd.read_csv(
        paths.feature_descriptions,
        encoding=(
            UNSW_FEATURE_DESCRIPTION_ENCODING
        ),
        low_memory=False,
    )

    return UNSWNB15Data(
        training=training,
        testing=testing,
        feature_descriptions=(
            feature_descriptions
        ),
    )

def _validate_expected_count(
    value: object,
    name: str,
) -> int:
    """Validate one expected positive row count."""

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    count = int(value)

    if count <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return count


def _validate_curated_partition(
    frame: object,
    name: str,
    expected_rows: int,
) -> None:
    """Validate one curated UNSW-NB15 partition."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if frame.shape[0] != expected_rows:
        raise ValueError(
            f"{name} row count must be "
            f"{expected_rows}; received "
            f"{frame.shape[0]}."
        )

    if tuple(frame.columns) != (
        UNSW_CURATED_COLUMNS
    ):
        raise ValueError(
            f"{name} columns must exactly match "
            "UNSW_CURATED_COLUMNS."
        )

    if frame.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate rows."
        )

    if frame.isna().any().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    if frame["id"].duplicated().any():
        raise ValueError(
            f"{name} contains duplicate IDs."
        )

    numeric_columns = tuple(
        column
        for column in UNSW_CURATED_COLUMNS
        if column
        not in (
            *UNSW_CATEGORICAL_COLUMNS,
            "attack_cat",
        )
    )

    try:
        numeric_values = frame.loc[
            :,
            numeric_columns,
        ].to_numpy(
            dtype="float64",
        )
    except (TypeError, ValueError) as exception:
        raise TypeError(
            f"{name} numeric columns must be "
            "numeric."
        ) from exception

    if not pd.api.types.is_numeric_dtype(
        frame["id"]
    ):
        raise TypeError(
            f"{name} IDs must be numeric."
        )

    if not np.isfinite(
        numeric_values
    ).all():
        raise ValueError(
            f"{name} contains nonfinite values."
        )

    labels = frame["label"].to_numpy()

    if set(labels.tolist()) != {0, 1}:
        raise ValueError(
            f"{name} labels must contain exactly "
            "0 and 1."
        )

    categories = (
        frame["attack_cat"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    normal_categories = categories.eq(
        "normal"
    )
    normal_labels = frame["label"].eq(0)

    category_label_mismatches = (
        normal_categories.to_numpy(
            dtype=bool,
        )
        != normal_labels.to_numpy(
            dtype=bool,
        )
    )

    if category_label_mismatches.any():
        raise ValueError(
            f"{name} attack categories and "
            "labels are inconsistent."
        )


def validate_unsw_nb15(
    data: object,
    *,
    expected_training_rows: int = (
        UNSW_EXPECTED_TRAINING_ROWS
    ),
    expected_testing_rows: int = (
        UNSW_EXPECTED_TESTING_ROWS
    ),
    expected_feature_description_rows: int = (
        UNSW_EXPECTED_FEATURE_DESCRIPTION_ROWS
    ),
) -> UNSWNB15Data:
    """Validate loaded official UNSW-NB15 data."""

    if not isinstance(data, UNSWNB15Data):
        raise TypeError(
            "data must be a UNSWNB15Data instance."
        )

    training_rows = _validate_expected_count(
        expected_training_rows,
        "expected_training_rows",
    )
    testing_rows = _validate_expected_count(
        expected_testing_rows,
        "expected_testing_rows",
    )
    description_rows = _validate_expected_count(
        expected_feature_description_rows,
        "expected_feature_description_rows",
    )

    _validate_curated_partition(
        data.training,
        "training",
        training_rows,
    )
    _validate_curated_partition(
        data.testing,
        "testing",
        testing_rows,
    )

    descriptions = data.feature_descriptions

    if not isinstance(
        descriptions,
        pd.DataFrame,
    ):
        raise TypeError(
            "feature_descriptions must be a "
            "pandas DataFrame."
        )

    if descriptions.shape[0] != description_rows:
        raise ValueError(
            "feature_descriptions row count must "
            f"be {description_rows}; received "
            f"{descriptions.shape[0]}."
        )

    if tuple(descriptions.columns) != (
        UNSW_FEATURE_DESCRIPTION_COLUMNS
    ):
        raise ValueError(
            "feature_descriptions columns must "
            "exactly match "
            "UNSW_FEATURE_DESCRIPTION_COLUMNS."
        )

    if descriptions.isna().any().any():
        raise ValueError(
            "feature_descriptions contains "
            "missing values."
        )

    return data

def _sha256_file(
    file_path: Path,
) -> str:
    """Calculate a file hash without loading it whole."""

    digest = sha256()

    with file_path.open("rb") as file_handle:
        while chunk := file_handle.read(
            1024 * 1024
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _sorted_string_counts(
    values: pd.Series,
) -> dict[str, int]:
    """Return deterministic JSON-safe counts."""

    counts = values.value_counts(
        dropna=False,
    ).sort_index()

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def _curated_file_evidence(
    file_path: Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Build evidence for one curated CSV."""

    return {
        "filename": file_path.name,
        "bytes": int(
            file_path.stat().st_size
        ),
        "sha256": _sha256_file(
            file_path
        ),
        "encoding": (
            UNSW_CURATED_FILE_ENCODING
        ),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "label_counts": (
            _sorted_string_counts(
                frame["label"]
            )
        ),
        "attack_category_counts": (
            _sorted_string_counts(
                frame["attack_cat"]
            )
        ),
    }


def build_unsw_nb15_manifest(
    paths: UNSWNB15Paths,
    data: UNSWNB15Data,
    *,
    expected_training_rows: int = (
        UNSW_EXPECTED_TRAINING_ROWS
    ),
    expected_testing_rows: int = (
        UNSW_EXPECTED_TESTING_ROWS
    ),
    expected_feature_description_rows: int = (
        UNSW_EXPECTED_FEATURE_DESCRIPTION_ROWS
    ),
) -> dict[str, object]:
    """Build deterministic raw-data provenance."""

    if not isinstance(paths, UNSWNB15Paths):
        raise TypeError(
            "paths must be a UNSWNB15Paths "
            "instance."
        )

    validated = validate_unsw_nb15(
        data,
        expected_training_rows=(
            expected_training_rows
        ),
        expected_testing_rows=(
            expected_testing_rows
        ),
        expected_feature_description_rows=(
            expected_feature_description_rows
        ),
    )

    description_rows = int(
        validated.feature_descriptions.shape[0]
    )

    return {
        "dataset": {
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
        },
        "hash_algorithm": "sha256",
        "validation": {
            "status": "passed",
            "curated_schema_columns": len(
                UNSW_CURATED_COLUMNS
            ),
            "feature_description_rows": (
                description_rows
            ),
            "identifier_scope": (
                "partition_local"
            ),
            "record_key": [
                "source_partition",
                "id",
            ],
        },
        "files": {
            "training": (
                _curated_file_evidence(
                    paths.training,
                    validated.training,
                )
            ),
            "testing": (
                _curated_file_evidence(
                    paths.testing,
                    validated.testing,
                )
            ),
            "feature_descriptions": {
                "filename": (
                    paths
                    .feature_descriptions
                    .name
                ),
                "bytes": int(
                    paths
                    .feature_descriptions
                    .stat()
                    .st_size
                ),
                "sha256": _sha256_file(
                    paths.feature_descriptions
                ),
                "encoding": (
                    UNSW_FEATURE_DESCRIPTION_ENCODING
                ),
                "rows": description_rows,
                "columns": int(
                    validated
                    .feature_descriptions
                    .shape[1]
                ),
            },
        },
    }

def write_unsw_nb15_manifest(
    paths: UNSWNB15Paths,
    data: UNSWNB15Data,
    *,
    output_path: str | Path = (
        DEFAULT_UNSW_MANIFEST_PATH
    ),
    expected_training_rows: int = (
        UNSW_EXPECTED_TRAINING_ROWS
    ),
    expected_testing_rows: int = (
        UNSW_EXPECTED_TESTING_ROWS
    ),
    expected_feature_description_rows: int = (
        UNSW_EXPECTED_FEATURE_DESCRIPTION_ROWS
    ),
) -> Path:
    """Write deterministic tracked provenance JSON."""

    if not isinstance(
        output_path,
        (str, Path),
    ):
        raise TypeError(
            "output_path must be a string or Path."
        )

    manifest = build_unsw_nb15_manifest(
        paths,
        data,
        expected_training_rows=(
            expected_training_rows
        ),
        expected_testing_rows=(
            expected_testing_rows
        ),
        expected_feature_description_rows=(
            expected_feature_description_rows
        ),
    )

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return destination
