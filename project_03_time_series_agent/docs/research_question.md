# Research Question

## Project title

Forecasting and Residual-Anomaly Agent for Hourly Bicycle Demand

## Primary research question

How accurately can hourly bicycle demand be forecast using chronological, leakage-safe evaluation, and how can forecast residuals be converted into transparent candidate anomaly alerts and an evidence-based operational model recommendation?

## Supporting questions

1. Do trained forecasting models outperform transparent baseline methods?
2. How stable are model rankings across chronological validation folds?
3. Which observations have unusually large out-of-sample forecast residuals?
4. How sensitive are anomaly findings to alternative scoring thresholds?
5. Can model selection be converted into a reproducible and auditable operational recommendation?

## Implemented forecasting scope

The final comparison contains six forecasting methods:

- historical-mean baseline;
- previous-hour naive baseline;
- 24-hour seasonal-naive baseline;
- 168-hour seasonal-naive baseline;
- Holt-Winters forecasting with daily seasonality;
- recursive Gradient Boosting forecasting.

Early planning considered ARIMA and SARIMA models. They were not retained in the final implementation. The completed scope instead prioritizes transparent baselines, classical exponential smoothing, and nonlinear machine learning under one chronological evaluation framework.

## Evaluation design

Models are evaluated without random train-test shuffling.

The project uses:

- a chronological holdout;
- 12 expanding weekly validation folds;
- a 168-hour forecast horizon per fold;
- identical validation periods for comparable models;
- 2,016 out-of-sample predictions per model;
- MAE, RMSE, sMAPE, and MASE;
- fold-level ranking and variability evidence;
- explicit tracking of negative count forecasts.

## Forecasting findings

Recursive Gradient Boosting achieved:

- mean MAE: 404.72;
- mean RMSE: 544.54;
- mean-MAE improvement over weekly seasonal naive: 3.77%;
- mean-RMSE improvement over weekly seasonal naive: 11.95%.

The weekly seasonal-naive benchmark achieved:

- mean MAE: 420.56;
- mean RMSE: 618.48.

The candidate model did not dominate every stability measure. Its fold-to-fold MAE standard deviation was 230.61, compared with 223.24 for the weekly benchmark. It recorded three MAE fold wins, while the benchmark recorded four. It also produced 199 negative raw forecasts before nonnegative constraints were applied.

These results support a qualified operational recommendation rather than a claim of universal superiority.

## Residual-anomaly findings

The anomaly analysis uses out-of-sample residuals so that candidate alerts are not derived from in-sample fitted errors.

Among 2,016 evaluated hours:

- 182 statistical anomalies were identified;
- 115 remained actionable after documented closures were separated;
- 87 actionable anomalies were positive demand surprises;
- 28 were negative demand surprises;
- the actionable candidate rate was 6.50%;
- the actionable hours formed 37 consecutive episodes.

The dataset contains no verified anomaly labels. These findings are therefore candidate alerts requiring contextual or domain review, not confirmed anomalies or causal conclusions.

## Recommendation outcome

The recommendation policy requires:

1. at least 2% mean-MAE improvement;
2. no degradation in mean RMSE;
3. evaluation over the same chronological folds.

Recursive Gradient Boosting passed all three gates and was selected as the preferred model. Weekly seasonal naive remains the operational fallback because it is simple, stable, naturally nonnegative, and resistant to recursive forecast-floor failures.

## Research-question status

| Question | Status | Evidence |
|---|---|---|
| Trained-model improvement | Answered | Gradient Boosting improved mean MAE by 3.77% and mean RMSE by 11.95% |
| Ranking stability | Answered | Fold wins and fold-level variability show that performance is not uniformly dominant |
| Residual anomalies | Answered | 115 actionable candidate hours were organized into 37 episodes |
| Alternative-threshold sensitivity | Not formally evaluated | Retained as future work and a limitation |
| Transparent recommendation | Answered | A three-gate policy selected a preferred model and fallback |

## Completion status

Data validation, preprocessing, feature engineering, forecasting, chronological evaluation, residual analysis, anomaly reporting, model recommendation, unified CLI workflows, and continuous integration are complete.

The implementation is validated by 158 automated tests, 90.07% package coverage, compilation checks, dependency checks, command-line smoke tests, and successful GitHub Actions jobs on Python 3.11 and 3.12.