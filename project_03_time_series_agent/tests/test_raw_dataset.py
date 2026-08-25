"""Integrity and structural tests for the raw time-series dataset."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SeoulBikeData.csv"

EXPECTED_SHA256 = (
    "373339B71A8935D69E9AF0ABF26A70744632119862EEB3919EFB389A7B749C60"
)

REQUIRED_COLUMNS = {
    "Date",
    "Hour",
    "Rented Bike Count",
    "Functioning Day",
}


@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    """Load the raw dataset once for all tests in this module."""
    return pd.read_csv(DATA_PATH, encoding="latin-1")


@pytest.fixture(scope="module")
def timestamps(raw_data: pd.DataFrame) -> pd.Series:
    """Construct hourly timestamps from the raw date and hour columns."""
    dates = pd.to_datetime(
        raw_data["Date"],
        format="%d/%m/%Y",
        errors="raise",
    )
    hours = pd.to_numeric(raw_data["Hour"], errors="raise")

    return dates + pd.to_timedelta(hours, unit="h")


def test_raw_file_exists() -> None:
    """The documented raw-data file should exist."""
    assert DATA_PATH.is_file()


def test_raw_file_checksum() -> None:
    """The raw file should match the documented digital fingerprint."""
    actual_hash = sha256(DATA_PATH.read_bytes()).hexdigest().upper()
    assert actual_hash == EXPECTED_SHA256


def test_raw_dataset_shape_and_columns(raw_data: pd.DataFrame) -> None:
    """The dataset should have its documented shape and key columns."""
    assert raw_data.shape == (8760, 14)
    assert REQUIRED_COLUMNS.issubset(raw_data.columns)


def test_raw_dataset_has_no_missing_cells(raw_data: pd.DataFrame) -> None:
    """The raw dataset should contain no missing cells."""
    assert int(raw_data.isna().sum().sum()) == 0


def test_timestamps_form_complete_hourly_series(
    timestamps: pd.Series,
) -> None:
    """Timestamps should be unique, ordered, and exactly hourly."""
    assert timestamps.is_monotonic_increasing
    assert not timestamps.duplicated().any()

    differences = timestamps.diff().dropna()
    expected_difference = pd.Timedelta(1, unit="h")

    assert differences.eq(expected_difference).all()
    assert timestamps.iloc[0] == pd.Timestamp("2017-12-01 00:00:00")
    assert timestamps.iloc[-1] == pd.Timestamp("2018-11-30 23:00:00")


def test_target_values_are_valid(raw_data: pd.DataFrame) -> None:
    """Rental counts should be numeric, present, and nonnegative."""
    target = pd.to_numeric(
        raw_data["Rented Bike Count"],
        errors="raise",
    )

    assert not target.isna().any()
    assert target.ge(0).all()
    assert int(target.eq(0).sum()) == 295


def test_zero_targets_are_documented_closures(
    raw_data: pd.DataFrame,
) -> None:
    """Every zero rental count should occur on a nonfunctioning day."""
    zero_target = raw_data["Rented Bike Count"].eq(0)
    nonfunctioning = raw_data["Functioning Day"].eq("No")

    assert zero_target.equals(nonfunctioning)