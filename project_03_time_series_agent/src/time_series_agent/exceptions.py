"""Custom exceptions used by the time-series agent."""


class TimeSeriesAgentError(Exception):
    """Base exception for errors raised by this project."""


class ConfigurationError(TimeSeriesAgentError):
    """Raised when project configuration is missing or invalid."""


class DataLoadError(TimeSeriesAgentError):
    """Raised when a dataset cannot be loaded."""


class DatasetNotFoundError(DataLoadError):
    """Raised when the configured dataset file does not exist."""


class MissingColumnsError(DataLoadError):
    """Raised when required dataset columns are missing."""


class DataParsingError(DataLoadError):
    """Raised when dates, hours, or target values cannot be parsed."""


class PreprocessingError(TimeSeriesAgentError):
    """Raised when time-series preprocessing cannot proceed safely."""