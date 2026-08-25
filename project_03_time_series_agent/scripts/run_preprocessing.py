"""Run the reproducible preprocessing workflow."""

import json

from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.exceptions import PreprocessingError
from time_series_agent.preprocessing import (
    preprocess_time_series,
    save_preprocessing_summary,
    save_processed_data,
)


CONFIG_PATH = "configs/default.yaml"
SUMMARY_PATH = "reports/validation/preprocessing_summary.json"


def main() -> None:
    """Load, validate, preprocess, save, and summarize the data."""
    config = load_data_config(CONFIG_PATH)

    if config.processed_file_path is None:
        raise PreprocessingError(
            "Configuration does not specify 'processed_file_path'."
        )

    loaded_data = load_time_series_csv(config)

    processed_data, summary = preprocess_time_series(
        data=loaded_data,
        config=config,
        closure_column="Functioning Day",
        closure_value="No",
    )

    save_processed_data(
        data=processed_data,
        output_path=config.processed_file_path,
    )
    save_preprocessing_summary(
        summary=summary,
        output_path=SUMMARY_PATH,
    )

    print(json.dumps(summary.to_dict(), indent=2))
    print(
        "Processed data saved to: "
        f"{config.processed_file_path}"
    )


if __name__ == "__main__":
    main()