"""Recursive machine-learning forecasting models."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from time_series_agent.baselines import BaseForecaster
from time_series_agent.exceptions import (
    FeatureEngineeringError,
    ForecastingError,
    ModelNotFittedError,
)
from time_series_agent.features import (
    build_lag_feature_set,
    build_recursive_feature_row,
)


@dataclass(frozen=True)
class GradientBoostingDiagnostics:
    """Diagnostics for a fitted recursive boosting model."""

    input_training_rows: int
    usable_training_rows: int
    dropped_training_rows: int
    feature_count: int
    feature_names: tuple[str, ...]
    training_mae: float
    residual_mean: float
    residual_standard_deviation: float
    feature_importances: dict[str, float]
    random_state: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable diagnostics dictionary."""
        return asdict(self)


class RecursiveGradientBoostingForecaster(
    BaseForecaster
):
    """Gradient Boosting with recursive multi-step prediction."""

    model_name = "gradient_boosting_recursive"

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 24, 168),
        rolling_windows: tuple[int, ...] = (24, 168),
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        clip_nonnegative: bool = True,
        frequency: str = "h",
    ) -> None:
        """Initialize an unfitted recursive boosting model."""
        if (
            not isinstance(n_estimators, int)
            or isinstance(n_estimators, bool)
            or n_estimators <= 0
        ):
            raise ForecastingError(
                "'n_estimators' must be a positive integer."
            )

        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
        ):
            raise ForecastingError(
                "'learning_rate' must be positive."
            )

        if (
            not isinstance(max_depth, int)
            or isinstance(max_depth, bool)
            or max_depth <= 0
        ):
            raise ForecastingError(
                "'max_depth' must be a positive integer."
            )

        if (
            not isinstance(min_samples_leaf, int)
            or isinstance(min_samples_leaf, bool)
            or min_samples_leaf <= 0
        ):
            raise ForecastingError(
                "'min_samples_leaf' must be a positive integer."
            )

        if not isinstance(random_state, int):
            raise ForecastingError(
                "'random_state' must be an integer."
            )

        if not isinstance(clip_nonnegative, bool):
            raise ForecastingError(
                "'clip_nonnegative' must be Boolean."
            )

        super().__init__(frequency=frequency)

        self.lags = lags
        self.rolling_windows = rolling_windows
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.clip_nonnegative = clip_nonnegative

        self._regressor: GradientBoostingRegressor | None = None
        self._training_history: pd.Series | None = None
        self._feature_names: tuple[str, ...] | None = None
        self._training_features: pd.DataFrame | None = None
        self._training_target: pd.Series | None = None
        self._last_raw_negative_forecast_count: int | None = None

    def _fit_values(
        self,
        training_series: pd.Series,
    ) -> None:
        """Construct features and fit Gradient Boosting."""
        minimum_rows = max(
            max(self.lags),
            max(self.rolling_windows),
        ) + 1

        if len(training_series) < minimum_rows:
            raise ForecastingError(
                f"Recursive Gradient Boosting requires at least "
                f"{minimum_rows} training observations."
            )

        training_data = pd.DataFrame(
            {
                "timestamp": training_series.index,
                "target": training_series.to_numpy(),
            }
        )

        try:
            feature_set = build_lag_feature_set(
                data=training_data,
                timestamp_column="timestamp",
                target_column="target",
                lags=self.lags,
                rolling_windows=self.rolling_windows,
            )
        except FeatureEngineeringError as error:
            raise ForecastingError(
                "Gradient Boosting feature construction failed."
            ) from error

        regressor = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

        regressor.fit(
            feature_set.features,
            feature_set.target,
        )

        self._regressor = regressor
        self._training_history = training_series.copy()
        self._feature_names = feature_set.feature_names
        self._training_features = (
            feature_set.features.copy()
        )
        self._training_target = feature_set.target.copy()

    def _forecast_values(
        self,
        horizon: int,
    ) -> np.ndarray:
        """Generate forecasts recursively using prior predictions."""
        if (
            self._regressor is None
            or self._training_history is None
            or self._feature_names is None
        ):
            raise ModelNotFittedError(
                "Recursive Gradient Boosting must be "
                "fitted before forecasting."
            )

        history = self._training_history.copy()
        predictions: list[float] = []
        raw_negative_count = 0

        for _ in range(horizon):
            future_timestamp = (
                history.index[-1] + self._offset
            )

            feature_row = build_recursive_feature_row(
                history=history,
                forecast_timestamp=future_timestamp,
                trend_index=len(history),
                lags=self.lags,
                rolling_windows=self.rolling_windows,
            )

            feature_row = feature_row.loc[
                :,
                list(self._feature_names),
            ]

            raw_prediction = float(
                self._regressor.predict(
                    feature_row
                )[0]
            )

            if not np.isfinite(raw_prediction):
                raise ForecastingError(
                    "Gradient Boosting produced a "
                    "nonfinite forecast."
                )

            if raw_prediction < 0:
                raw_negative_count += 1

            if self.clip_nonnegative:
                prediction = max(
                    0.0,
                    raw_prediction,
                )
            else:
                prediction = raw_prediction

            predictions.append(prediction)
            history.loc[future_timestamp] = prediction

        self._last_raw_negative_forecast_count = (
            raw_negative_count
        )

        return np.asarray(
            predictions,
            dtype="float64",
        )

    def last_raw_negative_forecast_count(self) -> int:
        """Return raw negative predictions in the last forecast."""
        if self._last_raw_negative_forecast_count is None:
            raise ModelNotFittedError(
                "A forecast must be generated before "
                "constraint diagnostics are requested."
            )

        return self._last_raw_negative_forecast_count

    def diagnostics(self) -> GradientBoostingDiagnostics:
        """Return training-fit and feature-importance diagnostics."""
        if (
            self._regressor is None
            or self._training_features is None
            or self._training_target is None
            or self._feature_names is None
            or self._training_history is None
        ):
            raise ModelNotFittedError(
                "Recursive Gradient Boosting must be "
                "fitted before diagnostics are requested."
            )

        fitted_values = self._regressor.predict(
            self._training_features
        )
        residuals = (
            self._training_target.to_numpy()
            - fitted_values
        )

        importance_pairs = sorted(
            zip(
                self._feature_names,
                self._regressor.feature_importances_,
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        feature_importances = {
            name: float(importance)
            for name, importance in importance_pairs
        }

        return GradientBoostingDiagnostics(
            input_training_rows=len(
                self._training_history
            ),
            usable_training_rows=len(
                self._training_features
            ),
            dropped_training_rows=(
                len(self._training_history)
                - len(self._training_features)
            ),
            feature_count=len(self._feature_names),
            feature_names=self._feature_names,
            training_mae=float(
                np.mean(np.abs(residuals))
            ),
            residual_mean=float(
                np.mean(residuals)
            ),
            residual_standard_deviation=float(
                np.std(residuals, ddof=1)
            ),
            feature_importances=feature_importances,
            random_state=self.random_state,
        )