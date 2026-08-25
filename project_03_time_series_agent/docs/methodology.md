# Methodology

## 1. Study design

This project develops a reproducible agent for hourly time-series
forecasting and residual-based anomaly detection.

The forecasting target is `Rented Bike Count` from the Seoul Bike
Sharing Demand dataset.

## 2. Data preservation

The original CSV is stored unchanged in `data/raw/`.

The raw-file SHA-256 checksum is:

`373339B71A8935D69E9AF0ABF26A70744632119862EEB3919EFB389A7B749C60`

Transformations are written to `data/processed/`. The raw file is never
overwritten.

## 3. Timestamp construction

The canonical timestamp is constructed as:

```text
timestamp = parsed Date + Hour

## 9. Baseline forecasting models

Four baseline models are implemented:

1. training-mean forecast;
2. last-value naïve forecast;
3. daily seasonal-naïve forecast with period 24;
4. weekly seasonal-naïve forecast with period 168.

All models require a chronologically ordered, regularly spaced
DatetimeIndex. They reject missing targets, temporal gaps, invalid
horizons, and prediction before fitting.

The baseline models are fitted using training observations only.
The generated next-24-hour preview is not an accuracy evaluation because
observed future values are unavailable.

Formal comparison will use chronological holdout and expanding-window
validation.

## 10. Chronological holdout evaluation

The final 168 hourly observations are reserved as a one-week test set.
All preceding observations form the training set.

Four baselines are fitted using training data only. Forecast accuracy is
measured using MAE, RMSE, sMAPE, and MASE.

The MASE denominator is calculated exclusively from training data using
a daily seasonal period of 24 hours.

Ordinary MAPE is not reported because the target includes zero values.

Results from this single holdout are provisional. Expanding-window
validation is required before making a model recommendation.