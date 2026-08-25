"""Load and parse raw time-series data."""

from pathlib import Path

import pandas as pd

from time_series_agent.config import DataConfig
from time_series_agent.exceptions import (
    DataLoadError,
    DataParsingError,
    DatasetNotFoundError,
    MissingColumnsError,
)


def _required_columns(config: DataConfig) -> set[str]:
    """Return the raw columns required by the data loader."""
    return {
        config.date_column,
        config.hour_column,
        config.target_column,
    }


def _parse_dates(
    data: pd.DataFrame,
    config: DataConfig,
) -> pd.Series:
    """Parse the configured date column."""
    try:
        parsed_dates = pd.to_datetime(
            data[config.date_column],
            format=config.date_format,
            errors="coerce",
        )
    except (TypeError, ValueError) as error:
        raise DataParsingError(
            f"Column '{config.date_column}' could not be parsed as dates."
        ) from error

    invalid_count = int(parsed_dates.isna().sum())

    if invalid_count:
        raise DataParsingError(
            f"Column '{config.date_column}' contains "
            f"{invalid_count} invalid or missing date value(s)."
        )

    return parsed_dates


def _parse_hours(
    data: pd.DataFrame,
    config: DataConfig,
) -> pd.Series:
    """Parse and validate integer hours from 0 through 23."""
    hours = pd.to_numeric(
        data[config.hour_column],
        errors="coerce",
    )

    invalid_count = int(hours.isna().sum())

    if invalid_count:
        raise DataParsingError(
            f"Column '{config.hour_column}' contains "
            f"{invalid_count} invalid or missing hour value(s)."
        )

    noninteger_count = int((hours % 1 != 0).sum())

    if noninteger_count:
        raise DataParsingError(
            f"Column '{config.hour_column}' contains "
            f"{noninteger_count} noninteger hour value(s)."
        )

    out_of_range_count = int((~hours.between(0, 23)).sum())

    if out_of_range_count:
        raise DataParsingError(
            f"Column '{config.hour_column}' contains "
            f"{out_of_range_count} value(s) outside the range 0-23."
        )

    return hours.astype("int64")


def _parse_target(
    data: pd.DataFrame,
    config: DataConfig,
) -> pd.Series:
    """Convert the configured forecasting target to numeric values."""
    target = pd.to_numeric(
        data[config.target_column],
        errors="coerce",
    )

    invalid_count = int(target.isna().sum())

    if invalid_count:
        raise DataParsingError(
            f"Column '{config.target_column}' contains "
            f"{invalid_count} invalid or missing numeric value(s)."
        )

    return target


def load_time_series_csv(config: DataConfig) -> pd.DataFrame:
    """Load raw CSV data and construct a timestamp column.

    The returned DataFrame retains its original row order. This function
    does not sort, impute, aggregate, remove, or otherwise clean rows.

    Parameters
    ----------
    config:
        Validated configuration describing the raw dataset.

    Returns
    -------
    pandas.DataFrame
        A copy of the raw data containing a constructed timestamp and
        parsed hour and target columns.

    Raises
    ------
    DatasetNotFoundError
        If the configured CSV file does not exist.
    MissingColumnsError
        If any required raw column is absent.
    DataParsingError
        If dates, hours, or target values are invalid.
    DataLoadError
        If pandas cannot read the CSV file.
    """
    path = Path(config.raw_file_path)

    if not path.is_file():
        raise DatasetNotFoundError(
            f"Dataset file does not exist: {path}"
        )

    try:
        data = pd.read_csv(path, encoding=config.encoding)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise DataLoadError(
            f"Dataset could not be read from: {path}"
        ) from error

    required = _required_columns(config)
    missing = required.difference(data.columns)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise MissingColumnsError(
            f"Dataset is missing required columns: {missing_text}"
        )

    parsed_dates = _parse_dates(data, config)
    parsed_hours = _parse_hours(data, config)
    parsed_target = _parse_target(data, config)

    loaded = data.copy()
    loaded[config.hour_column] = parsed_hours
    loaded[config.target_column] = parsed_target
    loaded[config.timestamp_column] = (
        parsed_dates
        + pd.to_timedelta(parsed_hours, unit="h")
    )

    return loaded