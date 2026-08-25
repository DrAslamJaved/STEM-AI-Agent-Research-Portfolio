"""Classical statistical forecasting models."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from time_series_agent.baselines import BaseForecaster
from time_series_agent.exceptions import (
    ForecastingError,
    ModelNotFittedError,
)


@dataclass(frozen=True)
class HoltWintersDiagnostics:
    """Diagnostics obtained from one fitted Holt-Winters model."""

    training_rows: int
    seasonal_period: int
    sum_squared_errors: float
    aic: float
    bic: float
    residual_mean: float
    residual_standard_deviation: float
    residual_lag_1_autocorrelation: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostics dictionary."""
        return asdict(self)


class HoltWintersForecaster(BaseForecaster):
    """Additive Holt-Winters model with damped additive trend."""

    model_name = "holt_winters"

    def __init__(
        self,
        seasonal_period: int = 24,
        damped_trend: bool = True,
        frequency: str = "h",
    ) -> None:
        """Initialize an unfitted Holt-Winters model."""
        if (
            not isinstance(seasonal_period, int)
            or isinstance(seasonal_period, bool)
            or seasonal_period < 2
        ):
            raise ForecastingError(
                "Holt-Winters seasonal period must be "
                "an integer of at least 2."
            )

        if not isinstance(damped_trend, bool):
            raise ForecastingError(
                "'damped_trend' must be Boolean."
            )

        super().__init__(frequency=frequency)

        self.seasonal_period = seasonal_period
        self.damped_trend = damped_trend
        self._fitted_result: Any | None = None
        self._training_index: pd.DatetimeIndex | None = None

    def _fit_values(self, training_series: pd.Series) -> None:
        """Fit an additive seasonal exponential-smoothing model."""
        minimum_rows = 2 * self.seasonal_period

        if len(training_series) < minimum_rows:
            raise ForecastingError(
                f"Holt-Winters requires at least {minimum_rows} "
                "training observations for two seasonal cycles."
            )

        try:
            model = ExponentialSmoothing(
                training_series,
                trend="add",
                damped_trend=self.damped_trend,
                seasonal="add",
                seasonal_periods=self.seasonal_period,
                initialization_method="estimated",
            )

            fitted_result = model.fit(
                optimized=True,
                remove_bias=False,
            )
        except (
            ValueError,
            TypeError,
            np.linalg.LinAlgError,
        ) as error:
            raise ForecastingError(
                "Holt-Winters model fitting failed."
            ) from error

        fitted_values = np.asarray(
            fitted_result.fittedvalues,
            dtype="float64",
        )

        if not np.isfinite(fitted_values).all():
            raise ForecastingError(
                "Holt-Winters produced nonfinite fitted values."
            )

        self._fitted_result = fitted_result
        self._training_index = training_series.index.copy()

    def _forecast_values(self, horizon: int) -> np.ndarray:
        """Generate future values from the fitted model."""
        if self._fitted_result is None:
            raise ModelNotFittedError(
                "Holt-Winters must be fitted before forecasting."
            )

        forecast = np.asarray(
            self._fitted_result.forecast(horizon),
            dtype="float64",
        )

        if not np.isfinite(forecast).all():
            raise ForecastingError(
                "Holt-Winters produced nonfinite forecasts."
            )

        return forecast

    def residuals(self) -> pd.Series:
        """Return fitted in-sample residuals."""
        if (
            self._fitted_result is None
            or self._training_index is None
        ):
            raise ModelNotFittedError(
                "Holt-Winters must be fitted before "
                "residuals are requested."
            )

        return pd.Series(
            np.asarray(
                self._fitted_result.resid,
                dtype="float64",
            ),
            index=self._training_index,
            name="holt_winters_residual",
        )

    def diagnostics(self) -> HoltWintersDiagnostics:
        """Return basic fit and residual diagnostics."""
        if self._fitted_result is None:
            raise ModelNotFittedError(
                "Holt-Winters must be fitted before "
                "diagnostics are requested."
            )

        residuals = self.residuals()

        return HoltWintersDiagnostics(
            training_rows=len(residuals),
            seasonal_period=self.seasonal_period,
            sum_squared_errors=float(
                self._fitted_result.sse
            ),
            aic=float(self._fitted_result.aic),
            bic=float(self._fitted_result.bic),
            residual_mean=float(residuals.mean()),
            residual_standard_deviation=float(
                residuals.std()
            ),
            residual_lag_1_autocorrelation=float(
                residuals.autocorr(lag=1)
            ),
        )