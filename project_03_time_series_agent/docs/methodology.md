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

## 11. Expanding-window validation

Baseline models are evaluated across 12 consecutive weekly test folds.

The first fold uses 6,744 training observations and 168 future test
observations. After every fold, the training window expands by 168
observations. Test observations always occur strictly after training
observations.

MAE, RMSE, sMAPE, and MASE are calculated separately for every
model-fold pair. Results are summarized using the mean and standard
deviation across folds.

The number of folds won by each model according to MAE is also reported.
This prevents model selection from relying only on one average value.

## 12. Holt-Winters forecasting

An additive Holt-Winters model is implemented using:

- additive level;
- additive damped trend;
- additive daily seasonality;
- seasonal period 24.

Additive seasonality is used because the target contains documented zero
values. Multiplicative seasonal models are not appropriate when observed
values can equal zero.

The model requires at least two complete seasonal cycles. Fitting
produces residuals, SSE, AIC, BIC, residual mean, residual standard
deviation, and lag-1 residual autocorrelation.

The next-24-hour forecast is an unevaluated preview. Model comparison
will use the same chronological and expanding-window design as the
baseline evaluation.

Holt-Winters can produce mathematically valid but physically impossible
negative forecasts for count data. The implementation therefore applies
a transparent nonnegative constraint:

\[
\hat y_t^* = \max(0,\hat y_t).
\]

The number of raw negative forecasts is recorded before clipping. This
constraint is applied identically during holdout and expanding-window
evaluation.

## 13. Holt-Winters expanding-window evaluation

Holt-Winters is evaluated using the same 12 expanding weekly folds used
for the four baselines. Every fold fits a new model using only the
training observations available at that time.

Forecasts are constrained to nonnegative counts. The number of raw
negative predictions is recorded separately for each fold before
clipping.

Model comparison considers mean error, error variability, fold wins,
and improvement relative to the weekly seasonal-naïve benchmark.

## 14. Leakage-safe feature engineering

Machine-learning features include target lags 1, 24, and 168; rolling
means and standard deviations over the preceding 24 and 168 hours;
cyclical hour and weekday encodings; and a deterministic trend index.

Every target-derived feature for time t uses observations no later than
time t-1. The current target is excluded from its own feature row.

Future weather measurements are not used because they would be unknown
unless separately forecast or provided as reliable external forecasts.

## 15. Recursive Gradient Boosting

A Gradient Boosting regressor is trained using lagged targets, shifted
rolling statistics, cyclical calendar features, and a trend index.

Multi-step forecasting is recursive. After each future hour is
predicted, that prediction is appended to the available history and may
be used in later lag and rolling features.

The model never uses hidden future targets. Weather measurements are
excluded because their future values are not assumed known.

The implementation uses a fixed random seed of 42 and constrains
predicted counts to nonnegative values.