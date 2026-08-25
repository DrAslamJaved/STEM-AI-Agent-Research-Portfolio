"""Inspect the raw Seoul Bike Sharing dataset without modifying it."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/SeoulBikeData.csv")
ENCODING = "latin-1"
DATE_FORMAT = "%d/%m/%Y"


def main() -> None:
    """Print a reproducible structural audit of the raw dataset."""
    print("RAW DATASET AUDIT")
    print("=" * 60)

    print(f"File exists: {DATA_PATH.exists()}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, encoding=ENCODING)

    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Duplicate complete rows: {data.duplicated().sum()}")
    print(f"Total missing cells: {data.isna().sum().sum()}")

    required_columns = {
        "Date",
        "Hour",
        "Rented Bike Count",
    }

    missing_required = required_columns.difference(data.columns)
    print(f"Missing required columns: {sorted(missing_required)}")

    if missing_required:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_required)}"
        )

    parsed_dates = pd.to_datetime(
        data["Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )

    invalid_dates = int(parsed_dates.isna().sum())
    print(f"Invalid dates: {invalid_dates}")

    numeric_hours = pd.to_numeric(data["Hour"], errors="coerce")
    invalid_hours = int(numeric_hours.isna().sum())
    out_of_range_hours = int((~numeric_hours.between(0, 23)).sum())

    print(f"Invalid hour values: {invalid_hours}")
    print(f"Hours outside 0-23: {out_of_range_hours}")

    timestamps = parsed_dates + pd.to_timedelta(numeric_hours, unit="h")

    print(f"Invalid constructed timestamps: {timestamps.isna().sum()}")
    print(f"First timestamp: {timestamps.min()}")
    print(f"Last timestamp: {timestamps.max()}")
    print(f"Timestamps sorted: {timestamps.is_monotonic_increasing}")
    print(f"Duplicate timestamps: {timestamps.duplicated().sum()}")

    sorted_timestamps = timestamps.sort_values().reset_index(drop=True)
    time_differences = sorted_timestamps.diff().dropna()

    expected_difference = pd.Timedelta(1, unit="h")
    irregular_intervals = int(
        (time_differences != expected_difference).sum()
    )

    expected_index = pd.date_range(
        start=sorted_timestamps.min(),
        end=sorted_timestamps.max(),
        freq="h",
    )

    missing_timestamps = expected_index.difference(
        pd.DatetimeIndex(sorted_timestamps)
    )

    print(f"Most common interval: {time_differences.mode().iloc[0]}")
    print(f"Intervals not equal to one hour: {irregular_intervals}")
    print(f"Expected hourly timestamps: {len(expected_index)}")
    print(f"Missing hourly timestamps: {len(missing_timestamps)}")

    target = pd.to_numeric(
        data["Rented Bike Count"],
        errors="coerce",
    )

    print(f"Invalid target values: {target.isna().sum()}")
    print(f"Negative target values: {(target < 0).sum()}")
    print(f"Zero target values: {(target == 0).sum()}")
    print(f"Minimum target: {target.min()}")
    print(f"Maximum target: {target.max()}")
    print(f"Mean target: {target.mean():.3f}")
    print(f"Median target: {target.median():.3f}")

    print("Hour values:", sorted(numeric_hours.dropna().unique().tolist()))
    print("Season values:", sorted(data["Seasons"].dropna().unique()))
    print("Holiday values:", sorted(data["Holiday"].dropna().unique()))
    print(
        "Functioning Day values:",
        sorted(data["Functioning Day"].dropna().unique()),
    )


if __name__ == "__main__":
    main()