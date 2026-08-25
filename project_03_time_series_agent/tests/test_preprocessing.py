"""Tests for leakage-safe time-series preprocessing."""

import json
from pathlib import Path

import pandas as pd
import pytest

from time_series_agent.config import DataConfig
from time_series_agent.exceptions import PreprocessingError
from time_series_agent.preprocessing import (
    preprocess_time_series,
    save_preprocessing_summary,
    save_processed_data,
)


def make_config(expected_rows: int | None = 3) -> DataConfig:
    """Create a test configuration."""
    return DataConfig(
        raw_file_path=Path("unused.csv"),
        encoding="utf-8",
        date_column="Date",
        hour_column="Hour",
        timestamp_column="timestamp",
        target_column="target",
        date_format="%d/%m/%Y",
        expected_frequency="h",
        expected_rows=expected_rows,
    )


def make_valid_data() -> pd.DataFrame:
    """Create a valid three-hour series."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 00:00:00",
                periods=3,
                freq="h",
            ),
            "target": [10, 0, 15],
            "Functioning Day": ["Yes", "No", "Yes"],
        }
    )


def test_preprocessing_preserves_rows_and_adds_closure_flag() -> None:
    """Valid preprocessing should retain all observations."""
    source = make_valid_data()

    processed, summary = preprocess_time_series(
        source,
        make_config(),
        closure_column="Functioning Day",
        closure_value="No",
    )

    assert len(processed) == len(source)
    assert summary.row_count_preserved
    assert processed["is_known_closure"].tolist() == [
        False,
        True,
        False,
    ]
    assert summary.known_closure_count == 1
    assert summary.zero_target_count == 1


def test_preprocessing_does_not_modify_input_dataframe() -> None:
    """The original loaded DataFrame should remain unchanged."""
    source = make_valid_data()
    original = source.copy(deep=True)

    processed, _ = preprocess_time_series(
        source,
        make_config(),
        closure_column="Functioning Day",
        closure_value="No",
    )

    pd.testing.assert_frame_equal(source, original)
    assert "is_known_closure" not in source.columns
    assert "is_known_closure" in processed.columns


def test_preprocessing_sorts_warning_status_data() -> None:
    """A disordered but otherwise valid series should be sorted."""
    source = make_valid_data().iloc[[2, 0, 1]].reset_index(drop=True)

    processed, summary = preprocess_time_series(
        source,
        make_config(),
        closure_column="Functioning Day",
        closure_value="No",
    )

    assert not summary.timestamps_were_sorted
    assert summary.timestamps_sorted_after_processing
    assert processed["timestamp"].is_monotonic_increasing


def test_preprocessing_refuses_invalid_series() -> None:
    """Duplicate timestamps should prevent preprocessing."""
    source = make_valid_data()
    source.loc[2, "timestamp"] = source.loc[1, "timestamp"]

    with pytest.raises(
        PreprocessingError,
        match="refused invalid time series",
    ):
        preprocess_time_series(
            source,
            make_config(),
            closure_column="Functioning Day",
            closure_value="No",
        )


def test_missing_requested_closure_column_is_rejected() -> None:
    """A requested but unavailable closure column should fail."""
    source = make_valid_data().drop(columns="Functioning Day")

    with pytest.raises(
        PreprocessingError,
        match="Closure column does not exist",
    ):
        preprocess_time_series(
            source,
            make_config(),
            closure_column="Functioning Day",
            closure_value="No",
        )


def test_processed_data_and_summary_can_be_saved(
    tmp_path: Path,
) -> None:
    """Processed CSV and JSON summary should be reproducible."""
    source = make_valid_data()

    processed, summary = preprocess_time_series(
        source,
        make_config(),
        closure_column="Functioning Day",
        closure_value="No",
    )

    csv_path = tmp_path / "processed.csv"
    summary_path = tmp_path / "summary.json"

    save_processed_data(processed, csv_path)
    save_preprocessing_summary(summary, summary_path)

    reloaded = pd.read_csv(csv_path)
    saved_summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    assert len(reloaded) == 3
    assert "is_known_closure" in reloaded.columns
    assert saved_summary["row_count_preserved"] is True
    assert saved_summary["known_closure_count"] == 1