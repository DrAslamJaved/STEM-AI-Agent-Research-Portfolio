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
```

Dates are parsed using the documented day-first format. Hours must be
integers from 0 through 23. Invalid dates, hours, targets, or required
columns produce explicit exceptions rather than silent corrections.

## 4. Time-series validation

The validation agent checks row count, timestamp validity, chronological
order, duplicate timestamps, missing hourly observations, irregular
intervals, missing targets, negative targets, and structural zeros.

Validation produces both JSON and Markdown reports. A dataset receives
valid status only when all essential structural and target checks pass.

The 295 zero-demand observations coincide with documented
non-functioning days. They are retained because they represent genuine
service closures rather than missing measurements.

## 5. Leakage-safe preprocessing

Preprocessing creates a deep copy of the loaded data, sorts observations
chronologically, and adds an explicit known-closure indicator.

No row is deleted merely because its target is zero. The raw dataset is
never overwritten, and processed outputs are stored separately.

All preprocessing decisions are summarized in a machine-readable report.

## 6. Exploratory time-series analysis

Exploration includes descriptive statistics, rolling summaries, hourly
demand profiles, autocorrelation, partial autocorrelation, additive
daily-seasonal decomposition, and the Augmented Dickey-Fuller test.

The target exhibits strong short-term, daily, and weekly dependence.
Observed autocorrelations are approximately 0.903 at lag 1, 0.682 at
lag 24, and 0.661 at lag 168.

The Augmented Dickey-Fuller p-value is below 0.001. This rejects the
unit-root null hypothesis but does not imply that daily and weekly
seasonality are absent.

## 7. Forecast-accuracy measures

Forecasts are evaluated using:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- symmetric mean absolute percentage error (sMAPE);
- mean absolute scaled error (MASE).

Ordinary MAPE is excluded because the series contains legitimate zeros.

The MASE scale is computed only from training observations using a
daily seasonal period of 24 hours. Test observations never contribute
to metric scaling or model fitting.

## 8. Temporal evaluation safeguards

Random train-test splitting is not used. Every test observation occurs
strictly after its corresponding training observations.

Models are compared on identical chronological folds. A new model
instance is fitted separately in every fold, preventing fitted state
from leaking between evaluations.

Model selection considers average error, variability between folds,
fold wins, physically impossible forecasts, and model complexity.

## 9. Baseline forecasting models

Four baseline models are implemented:

1. training-mean forecast;
2. last-value naive forecast;
3. daily seasonal-naive forecast with period 24;
4. weekly seasonal-naive forecast with period 168.

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
and improvement relative to the weekly seasonal-naive benchmark.

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

## 16. Gradient Boosting expanding-window evaluation

Recursive Gradient Boosting is evaluated on the same 12 expanding
weekly folds used for the baseline and Holt-Winters models.

For each fold, feature construction and model fitting use only the
training period. The model then produces a recursive 168-hour forecast.
Previously predicted values may enter later lag and rolling features,
but hidden test targets are never used.

The six-model results are:

| Rank | Model | Mean MAE | MAE standard deviation | Mean RMSE | Fold wins |
|---:|---|---:|---:|---:|---:|
| 1 | Recursive Gradient Boosting | 404.72 | 230.61 | 544.54 | 3 |
| 2 | Weekly seasonal naive | 420.56 | 223.24 | 618.48 | 4 |
| 3 | Daily seasonal naive | 444.04 | 270.58 | 621.37 | 3 |
| 4 | Holt-Winters | 516.69 | 200.79 | 641.29 | 0 |
| 5 | Training mean | 523.50 | 145.92 | 632.75 | 1 |
| 6 | Last-value naive | 566.23 | 181.07 | 703.82 | 1 |

Gradient Boosting improves mean MAE over the weekly seasonal-naive
benchmark by 15.84 bikes, corresponding to 3.77%. It also produces the
lowest mean RMSE.

This advantage is not uniform across folds. Gradient Boosting wins 3
folds compared with 4 wins for the weekly benchmark, and its MAE
standard deviation is slightly larger.

The unconstrained Gradient Boosting forecasts include 199 negative
values among 2,016 test predictions. These values are recorded before
being constrained to zero. This corresponds to approximately 9.87% of
its raw forecasts.

The evidence supports Gradient Boosting as the preferred model according
to average predictive accuracy. Weekly seasonal naive is retained as a
strong fallback because it is simple, interpretable, competitive, and
naturally nonnegative.

Feature importance describes how the fitted model distributes predictive
importance; it must not be interpreted as evidence of a causal
relationship.