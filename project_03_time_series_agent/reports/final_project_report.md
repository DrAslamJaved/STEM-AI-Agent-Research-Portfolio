# Final Project Report and Reproducibility Audit

## Project 03 — Time-Series Forecasting and Residual-Anomaly Agent

- **Project owner:** Dr. Muhammad Aslam Javed
- **Dataset:** Seoul Bike Sharing Demand
- **Audit date:** 6 September 2026
- **Status:** Complete

## 1. Executive summary

This project develops a reproducible agent-assisted system for forecasting
hourly bicycle demand, identifying unusual out-of-sample forecast
residuals, grouping candidate anomalies into operational episodes, and
producing an evidence-based model recommendation.

Six forecasting methods were evaluated using chronological validation.
Recursive Gradient Boosting achieved the lowest mean MAE and RMSE. Its
mean MAE was 404.72, compared with 420.56 for the strongest transparent
benchmark, weekly seasonal naive. This represents a 3.77% MAE
improvement. Its mean RMSE improvement was 11.95%.

The improvement is meaningful but not uniformly dominant. Gradient
Boosting had slightly greater fold-to-fold MAE variability, won fewer
individual folds than the weekly benchmark, and produced 199 negative
raw forecasts before nonnegative constraints were applied.

The recommendation agent therefore selects recursive Gradient Boosting
as the preferred model while retaining weekly seasonal naive as the
operational fallback.

Residual analysis identified 115 actionable candidate anomaly hours,
organized into 37 consecutive episodes. Because the dataset contains no
verified anomaly labels, these observations are reported as candidates
requiring contextual review rather than confirmed anomalies.

The final implementation passes 158 automated tests with 90.07% package
coverage. It also passes dependency, compilation, command-line, raw-data
integrity, and continuous-integration checks on Python 3.11 and 3.12.

## 2. Research objective

The primary research question is:

> How accurately can hourly bicycle demand be forecast using
> chronological, leakage-safe evaluation, and how can forecast residuals
> be converted into transparent candidate anomaly alerts and an
> evidence-based operational model recommendation?

The project addresses five supporting concerns:

1. improvement over transparent baselines;
2. stability across chronological validation folds;
3. identification of unusual out-of-sample residuals;
4. sensitivity of anomaly conclusions;
5. transparent and reproducible operational recommendation.

Alternative-threshold sensitivity was not formally evaluated and remains
future work.

## 3. Dataset and provenance

The Seoul Bike Sharing Demand dataset contains:

- 8,760 hourly observations;
- 14 original columns;
- observations from 1 December 2017 through 30 November 2018;
- no missing hourly timestamps;
- no missing cells;
- no duplicate timestamps;
- no negative target values.

The forecasting target is `Rented Bike Count`.

Its observed properties include:

- minimum: 0;
- maximum: 3,556;
- mean: 704.602;
- median: 504.500;
- zero-demand observations: 295.

The dataset is distributed by the UCI Machine Learning Repository under
the Creative Commons Attribution 4.0 International licence:

> Seoul Bike Sharing Demand. (2020). UCI Machine Learning Repository.
>
> https://doi.org/10.24432/C5F62R

The raw source file is preserved byte-for-byte. Its SHA-256 checksum is:

`373339B71A8935D69E9AF0ABF26A70744632119862EEB3919EFB389A7B749C60`

A project-level `.gitattributes` rule prevents cross-platform
line-ending conversion of the raw CSV.

## 4. Preprocessing and feature engineering

Dates are parsed using the explicit `%d/%m/%Y` format and combined with
the validated hour column to create a chronological timestamp.

Structural preprocessing retains all 8,760 observations. No raw
observations are deleted or imputed.

Lagged and rolling predictors are created using historical observations
only. The maximum required history is 168 hours. Consequently, the
model-ready feature matrix contains 8,592 observations after excluding
the first 168 hours for which the full historical feature set is not
available.

The final feature matrix contains 12 predictors. All chronological
splitting occurs before any data-derived fitting operation that could
introduce temporal leakage.

## 5. Forecasting methods

The final comparison contains:

1. historical mean;
2. previous-hour naive;
3. 24-hour seasonal naive;
4. 168-hour seasonal naive;
5. Holt-Winters with daily seasonality;
6. recursive Gradient Boosting.

Early planning considered ARIMA and SARIMA. These models were not
retained in the final implementation. The completed scope instead
compares transparent baselines, classical exponential smoothing, and
nonlinear machine learning under a common evaluation design.

## 6. Chronological evaluation design

The project does not use random train-test shuffling.

Evaluation consists of:

- a chronological holdout;
- 12 expanding weekly validation folds;
- a 168-hour test horizon per fold;
- identical folds for comparable models;
- 2,016 out-of-sample predictions per model.

The principal metrics are:

- mean absolute error (MAE);
- root mean squared error (RMSE);
- symmetric mean absolute percentage error (sMAPE);
- mean absolute scaled error (MASE).

Fold-level wins, variability, and negative raw forecasts are retained as
supporting diagnostic evidence.

## 7. Model-comparison results

| Rank | Model | Mean MAE | MAE fold wins |
|---:|---|---:|---:|
| 1 | Recursive Gradient Boosting | 404.72 | 3 |
| 2 | Seasonal naive, 168 hours | 420.56 | 4 |
| 3 | Seasonal naive, 24 hours | 444.04 | 3 |
| 4 | Holt-Winters | 516.69 | 0 |
| 5 | Historical mean | 523.50 | 1 |
| 6 | Previous-hour naive | 566.23 | 1 |

The detailed preferred-model comparison is:

| Diagnostic | Gradient Boosting | Weekly seasonal naive |
|---|---:|---:|
| Mean MAE | 404.72 | 420.56 |
| Mean RMSE | 544.54 | 618.48 |
| MAE standard deviation | 230.61 | 223.24 |
| MAE fold wins | 3 | 4 |
| Raw negative forecasts | 199 | 0 |
| Raw negative forecast rate | 9.87% | 0.00% |
| Operational complexity | High | Low |

Relative improvement is calculated using:

\[
I =
100
\frac{
E_{\mathrm{benchmark}} -
E_{\mathrm{candidate}}
}{
E_{\mathrm{benchmark}}
}.
\]

This gives:

- mean-MAE improvement: 3.77%;
- mean-RMSE improvement: 11.95%.

These results show that Gradient Boosting has the best aggregate error
performance, but the weekly seasonal-naive model remains competitive and
more operationally stable.

## 8. Residual-anomaly analysis

Anomalies are detected from out-of-sample Gradient Boosting residuals,
not from in-sample fitted errors.

The analysis distinguishes documented nonfunctioning periods from
unexpected residual behavior.

| Anomaly diagnostic | Result |
|---|---:|
| Out-of-sample residuals | 2,016 |
| Reference observations | 1,769 |
| Documented closure observations | 247 |
| Statistical anomalies | 182 |
| Actionable candidate hours | 115 |
| Positive demand surprises | 87 |
| Negative demand surprises | 28 |
| Actionable candidate rate | 6.50% |
| Consecutive episodes | 37 |

Episode classification produced:

| Episode context | Episodes | Anomalous hours |
|---|---:|---:|
| Forecast-floor positive episode | 12 | 62 |
| Rain-coincident negative episode | 5 | 22 |
| Other residual episode | 20 | 31 |

The ten most concentrated dates contain 93 of the 115 actionable hours,
or 80.87%. Candidate alerts therefore occur mainly in connected episodes
rather than as independent isolated observations.

One important negative episode occurred on 5 October 2018. It contained
12 consecutive candidate anomaly hours during rainfall. Observed demand
totaled 396 rentals, compared with a forecast total of 20,521, while
accumulated rainfall was 31 mm.

A strong positive episode occurred on 17 September 2018 when the
recursive forecast approached its lower boundary while observed demand
remained high. This represents a forecast-floor and model-recovery
failure rather than sufficient evidence of an abnormal real-world event.

## 9. Model recommendation

The recommendation agent uses three explicit gates:

1. the candidate must improve mean MAE by at least 2%;
2. the candidate must not degrade mean RMSE;
3. candidate and benchmark must use the same chronological folds.

Recursive Gradient Boosting passed all three gates.

The final recommendation is:

- preferred model: `gradient_boosting_recursive`;
- fallback model: `seasonal_naive_168`;
- decision status: `candidate_selected`.

The preferred model should be accompanied by:

- nonnegative forecast constraints;
- monitoring of raw negative predictions;
- monitoring of forecast-floor behavior;
- failure handling for recursive prediction;
- fallback to weekly seasonal naive when candidate diagnostics fail.

The 2% improvement rule is an explicit project policy. It is not claimed
to be a universal statistical threshold.

## 10. Agentic system contribution

The project demonstrates agentic behavior through reproducible,
inspectable workflow components rather than opaque autonomous claims.

The system can:

- select and execute named workflows;
- preserve dependency order across 17 scripts;
- stop when a child process fails;
- preview execution safely using `--dry-run`;
- collect out-of-sample residual evidence;
- rank candidate anomalies;
- group consecutive anomaly episodes;
- attach contextual explanations;
- apply explicit model-selection gates;
- produce machine-readable and human-readable recommendations.

Human judgment remains necessary for:

- interpreting candidate anomalies;
- assessing causal explanations;
- approving operational deployment;
- changing decision thresholds;
- evaluating external validity.

## 11. Software and reproducibility validation

The final local audit produced:

| Validation check | Result |
|---|---|
| Dependency consistency | Passed |
| Automated tests | 158 passed |
| Package coverage | 90.07% |
| Required coverage | 90% |
| Source compilation | Passed |
| CLI help smoke test | Passed |
| Complete workflow dry run | 17 steps passed |
| Raw-data checksum tests | Passed |
| `git diff --check` | Passed |
| GitHub Actions, Python 3.11 | Passed |
| GitHub Actions, Python 3.12 | Passed |

The final JUnit evidence is stored in:

`reports/validation/phase_15_pytest.xml`

Continuous integration is defined in:

`.github/workflows/project_03_ci.yml`

## 12. Limitations and threats to validity

The main limitations are:

1. the dataset represents one city and approximately one year;
2. the source does not document a timezone;
3. observed rentals may be constrained by bicycle availability;
4. service closures create structural zeros;
5. future weather information is not supplied to the recursive forecast;
6. recursive prediction can accumulate error;
7. raw Gradient Boosting predictions can be negative;
8. no verified anomaly labels are available;
9. anomaly-threshold sensitivity was not formally evaluated;
10. no formal paired-error significance test was performed;
11. unrecorded events may explain some large residuals;
12. results may not generalize to other cities, years, or transport
    systems.

A large residual must not be interpreted automatically as fraud, system
failure, unusual human behaviour, or a confirmed real-world anomaly.

## 13. Research-question outcomes

| Research question | Outcome |
|---|---|
| Do trained models improve on baselines? | Yes, Gradient Boosting improved mean MAE by 3.77% and RMSE by 11.95% over the strongest baseline |
| Are rankings stable across folds? | Partly; aggregate performance favours Gradient Boosting, but the weekly benchmark has lower MAE variability and more fold wins |
| Can residuals identify unusual observations? | Yes, 115 actionable candidate hours were identified and grouped into 37 episodes |
| Are findings stable under alternative thresholds? | Not formally evaluated; retained as future work |
| Can the recommendation be transparent? | Yes; three explicit decision gates determine the preferred model and fallback |

## 14. Evidence map

| Evidence | Location |
|---|---|
| Research question | `docs/research_question.md` |
| Dataset provenance and limitations | `docs/dataset_card.md` |
| Complete methodology | `docs/methodology.md` |
| Baseline evaluation | `reports/metrics/baseline_expanding_summary.csv` |
| Machine-learning evaluation | `reports/metrics/ml_expanding_summary.csv` |
| Out-of-sample residuals | `reports/metrics/gradient_boosting_oos_residuals.csv` |
| Candidate anomaly labels | `reports/metrics/gradient_boosting_anomaly_labels.csv` |
| Episode evidence | `reports/metrics/anomaly_episodes.csv` |
| Anomaly interpretation | `reports/anomaly_report.md` |
| Recommendation data | `reports/metrics/model_recommendation.json` |
| Recommendation report | `reports/model_recommendation.md` |
| Final test evidence | `reports/validation/phase_15_pytest.xml` |
| CI workflow | `.github/workflows/project_03_ci.yml` |

## 15. Conclusion

The project provides a complete and reproducible workflow for hourly
demand forecasting, residual-based candidate anomaly detection, and
transparent model recommendation.

Recursive Gradient Boosting delivers the strongest aggregate predictive
performance, while weekly seasonal naive remains a credible operational
fallback. The anomaly system adds value by transforming isolated
residuals into contextual episodes, but its outputs remain candidate
alerts because verified anomaly labels are unavailable.

The project demonstrates chronological validation, leakage control,
baseline comparison, diagnostic transparency, tested software,
cross-platform raw-data integrity, command-line orchestration, and
continuous integration. These features make the work suitable as a
portfolio demonstration of rigorous time-series modelling and
agent-assisted scientific workflow design.