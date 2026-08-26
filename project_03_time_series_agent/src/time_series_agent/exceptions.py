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


class ExplorationError(TimeSeriesAgentError):
    """Raised when exploratory time-series analysis cannot proceed."""


class ForecastingError(TimeSeriesAgentError):
    """Raised when a forecasting operation is invalid."""


class ModelNotFittedError(ForecastingError):
    """Raised when prediction is requested before model fitting."""


class EvaluationError(TimeSeriesAgentError):
    """Raised when forecast evaluation cannot proceed safely."""


class FeatureEngineeringError(TimeSeriesAgentError):
    """Raised when leakage-safe features cannot be constructed."""


class AnomalyDetectionError(TimeSeriesAgentError):
    """Raised when residual anomaly detection cannot proceed safely."""


class RecommendationError(TimeSeriesAgentError):
    """Raised when model recommendation cannot proceed safely."""


class CliExecutionError(TimeSeriesAgentError):
    """Raised when a command-line workflow cannot complete."""