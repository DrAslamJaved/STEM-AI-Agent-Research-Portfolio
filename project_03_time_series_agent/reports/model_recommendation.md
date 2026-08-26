# Forecasting Model Recommendation

## Decision

**Preferred model:** `gradient_boosting_recursive`

**Fallback model:** `seasonal_naive_168`

**Decision status:** `candidate_selected`

gradient_boosting_recursive improves mean MAE by 3.77% and does not worsen mean RMSE relative to the benchmark.

## Selection policy

The agent ranks models by mean MAE across identical expanding-window folds. A more complex candidate replaces the weekly seasonal-naive benchmark only when:

1. its mean-MAE improvement is at least 2%;
2. its mean RMSE is no worse than the benchmark;
3. all models were evaluated over the same folds.

| Decision check | Requirement | Observed | Result |
|---|---:|---:|---|
| MAE improvement | >= 2.00% | 3.77% | Pass |
| RMSE improvement | No degradation | 11.95% | Pass |
| Comparable folds | Same fold count | Yes | Pass |

## Six-model evidence

| Rank | Model | Mean MAE | MAE standard deviation | Mean RMSE | Mean sMAPE | Mean MASE | Fold wins | Raw negative forecasts |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | gradient_boosting_recursive | 404.72 | 230.61 | 544.54 | 71.57 | 1.508 | 3 | 199 |
| 2 | seasonal_naive_168 | 420.56 | 223.24 | 618.48 | 71.80 | 1.573 | 4 | 0 |
| 3 | seasonal_naive_24 | 444.04 | 270.58 | 621.37 | 80.87 | 1.647 | 3 | 0 |
| 4 | holt_winters_24 | 516.69 | 200.79 | 641.29 | 86.35 | 1.922 | 0 | 224 |
| 5 | mean | 523.50 | 145.92 | 632.75 | 80.16 | 1.953 | 1 | 0 |
| 6 | naive | 566.23 | 181.07 | 703.82 | 96.08 | 2.099 | 1 | 0 |

## Candidate strengths

- Mean MAE improved by 3.77% relative to the weekly benchmark.
- Mean RMSE improved by 11.95%.
- The selected model ranked first by the primary metric, mean MAE.

## Candidate cautions

- The best-MAE candidate has greater fold-to-fold MAE variability than the benchmark.
- The best-MAE candidate wins fewer folds than the benchmark.
- The best-MAE candidate produces raw negative count forecasts and requires nonnegative clipping.
- Raw-negative forecast rate: 9.87%.
- Selected-model complexity: high.
- Fallback-model complexity: low.

## Anomaly-monitoring evidence

The residual agent identified 115 actionable candidate hours, corresponding to 6.50% of nonclosure residuals.

These alerts form 37 consecutive episodes.

- Forecast-floor positive anomaly hours: 62.
- Rain-coincident negative anomaly hours: 22.
- Other residual anomaly hours: 31.
- Known closures remain separate from actionable alerts: 247 closure hours.

Forecast-floor episodes demonstrate that the selected model can become trapped near zero during recursive multi-step prediction. This limitation supports retaining the weekly seasonal-naive fallback.

## Operational policy

1. Use recursive Gradient Boosting as the preferred forecasting model.
2. Apply and report the nonnegative forecast constraint.
3. Monitor raw-negative forecasts and forecast-floor episodes.
4. Use weekly seasonal naive when Gradient Boosting cannot fit, predict, or produce acceptable diagnostics.
5. Keep documented closures separate from unexpected anomaly alerts.
6. Treat residual anomalies as candidates requiring contextual or domain review.

## Limitations

- The recommendation is based on one public dataset and 12 weekly expanding-window folds.
- The 2% selection threshold is an explicit operational policy rather than a universal statistical law.
- No formal significance test has yet been applied to paired fold errors.
- The selected model excludes unknown future weather.
- The anomaly dataset contains no verified ground-truth anomaly labels.

## Machine-readable decision

The complete structured decision is stored in `reports/metrics/model_recommendation.json`.
