"""Allow execution with python -m time_series_agent."""

from time_series_agent.cli import main


if __name__ == "__main__":
    raise SystemExit(main())