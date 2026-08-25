# Research Question

## Project title

Project 03 — Time-Series Forecasting and Anomaly Detection Agent

## Primary research question

How accurately can classical and machine-learning time-series models
forecast future observations, and can out-of-sample forecast residuals
be used to identify statistically meaningful anomalies?

## Supporting questions

1. Do trained forecasting models outperform mean, naïve, and
   seasonal-naïve baselines?
2. How stable is forecasting performance across different
   time-series validation windows?
3. Which observations are identified as anomalies using forecast
   residuals?
4. Are detected anomalies stable under alternative anomaly thresholds?
5. Can an automated agent produce a transparent and reproducible
   model recommendation?

## Proposed model families

The project will investigate the following model families:

- mean forecast;
- naïve forecast;
- seasonal-naïve forecast;
- exponential smoothing or Holt-Winters;
- ARIMA or SARIMA;
- one lag-based machine-learning model.

The final model set may be adjusted according to the selected dataset's
frequency, size, trend, seasonality, and data quality.

## Proposed anomaly-detection methods

The project will initially compare:

- standardized residual or z-score detection;
- robust median absolute deviation detection.

Anomalies will preferably be calculated from out-of-sample forecast
residuals rather than only from fitted in-sample residuals.

## Evaluation strategy

Forecasts will be evaluated chronologically. Random train-test splitting
will not be used.

The evaluation will include:

- chronological holdout testing;
- expanding-window validation;
- comparison with simple baselines;
- MAE and RMSE;
- sMAPE where appropriate;
- MASE where a valid scaling denominator exists.

## Current status

The project foundation is under construction. No dataset has yet been
selected, and no empirical forecasting results are currently available.