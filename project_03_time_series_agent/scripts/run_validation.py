"""Run validation on the configured raw time-series dataset."""

import json

from time_series_agent.config import load_data_config
from time_series_agent.data_loader import load_time_series_csv
from time_series_agent.validation import (
    save_validation_report,
    validate_time_series,
)


CONFIG_PATH = "configs/default.yaml"
JSON_REPORT_PATH = "reports/validation/time_series_validation.json"
MARKDOWN_REPORT_PATH = "reports/validation/time_series_validation.md"


def main() -> None:
    """Load, validate, save, and display the validation report."""
    config = load_data_config(CONFIG_PATH)
    data = load_time_series_csv(config)

    report = validate_time_series(
        data=data,
        config=config,
        closure_column="Functioning Day",
        closure_value="No",
    )

    save_validation_report(
        report=report,
        json_path=JSON_REPORT_PATH,
        markdown_path=MARKDOWN_REPORT_PATH,
    )

    print(json.dumps(report.to_dict(), indent=2))

    if report.status == "invalid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()