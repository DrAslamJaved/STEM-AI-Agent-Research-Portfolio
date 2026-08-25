"""Prepare validated time-series data for analysis and modelling."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from time_series_agent.config import DataConfig
from time_series_agent.exceptions import PreprocessingError
from time_series_agent.validation import validate_time_series


@dataclass(frozen=True)
class PreprocessingSummary:
    """Summary of transparent preprocessing operations."""

    validation_status: str
    input_rows: int
    output_rows: int
    row_count_preserved: bool
    timestamps_were_sorted: bool
    timestamps_sorted_after_processing: bool
    closure_indicator_column: str
    known_closure_count: int
    zero_target_count: int
    missing_target_count: int
    duplicate_timestamp_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def preprocess_time_series(
    data: pd.DataFrame,
    config: DataConfig,
    closure_column: str | None = None,
    closure_value: str | None = None,
) -> tuple[pd.DataFrame, PreprocessingSummary]:
    """Prepare validated data without deleting or imputing observations.

    The function:

    1. validates the supplied data;
    2. refuses data with an invalid validation status;
    3. makes a deep copy;
    4. sorts the copy chronologically;
    5. adds an ``is_known_closure`` Boolean column;
    6. preserves every input row.

    Parameters
    ----------
    data:
        Loaded time-series data.
    config:
        Data-loading configuration.
    closure_column:
        Optional source column identifying known closures.
    closure_value:
        Value representing a closure.

    Returns
    -------
    tuple[pandas.DataFrame, PreprocessingSummary]
        Processed data and a summary of the operations.

    Raises
    ------
    PreprocessingError
        If validation fails or closure information is inconsistent
        with the requested preprocessing configuration.
    """
    report = validate_time_series(
        data=data,
        config=config,
        closure_column=closure_column,
        closure_value=closure_value,
    )

    if report.status == "invalid":
        error_text = "; ".join(report.errors)
        raise PreprocessingError(
            f"Preprocessing refused invalid time series: {error_text}"
        )

    if (
        closure_column is not None
        and closure_value is not None
        and closure_column not in data.columns
    ):
        raise PreprocessingError(
            f"Closure column does not exist: {closure_column}"
        )

    input_rows = len(data)
    timestamps_were_sorted = bool(
        data[config.timestamp_column].is_monotonic_increasing
    )

    processed = data.copy(deep=True)

    processed = processed.sort_values(
        by=config.timestamp_column,
        kind="stable",
    ).reset_index(drop=True)

    if closure_column is not None and closure_value is not None:
        processed["is_known_closure"] = (
            processed[closure_column].eq(closure_value)
        )
    else:
        processed["is_known_closure"] = False

    output_rows = len(processed)

    summary = PreprocessingSummary(
        validation_status=report.status,
        input_rows=input_rows,
        output_rows=output_rows,
        row_count_preserved=(input_rows == output_rows),
        timestamps_were_sorted=timestamps_were_sorted,
        timestamps_sorted_after_processing=bool(
            processed[
                config.timestamp_column
            ].is_monotonic_increasing
        ),
        closure_indicator_column="is_known_closure",
        known_closure_count=int(
            processed["is_known_closure"].sum()
        ),
        zero_target_count=int(
            processed[config.target_column].eq(0).sum()
        ),
        missing_target_count=int(
            processed[config.target_column].isna().sum()
        ),
        duplicate_timestamp_count=int(
            processed[
                config.timestamp_column
            ].duplicated().sum()
        ),
    )

    return processed, summary


def save_processed_data(
    data: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save processed data without including a DataFrame index."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def save_preprocessing_summary(
    summary: PreprocessingSummary,
    output_path: str | Path,
) -> None:
    """Save a preprocessing summary as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )