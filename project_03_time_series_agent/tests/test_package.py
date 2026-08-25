"""Tests for the time_series_agent package."""

import time_series_agent


def test_package_has_expected_version() -> None:
    """The package should expose its current version."""
    assert time_series_agent.__version__ == "0.1.0"