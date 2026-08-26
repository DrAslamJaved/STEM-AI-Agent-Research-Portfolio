# Project 03 — Time-Series Forecasting Agent

A reproducible Python project for time-series forecasting, time-aware
model evaluation, residual-based anomaly detection, and transparent
model recommendation.

## Research objective

This project investigates how accurately classical and machine-learning
models can forecast future observations and whether out-of-sample
forecast residuals can identify meaningful anomalies.

The full research question is documented in
[`docs/research_question.md`](docs/research_question.md).

## Planned workflow

1. Select and document a public time-series dataset.
2. Load and validate the data.
3. Examine trend, seasonality, stationarity, and autocorrelation.
4. Construct mean, naive, and seasonal-naive baselines.
5. Train classical forecasting models.
6. Train one lag-based machine-learning model.
7. Evaluate models using chronological validation.
8. Detect anomalies from forecast residuals.
9. recommend a model using quantitative evidence.
10. Generate reproducible reports.

## Dataset

This project uses the Seoul Bike Sharing Demand dataset from the
UCI Machine Learning Repository.

The dataset contains 8,760 consecutive hourly observations from
1 December 2017 through 30 November 2018. The forecasting target is
`Rented Bike Count`.

The raw data contains:

- no missing cells;
- no duplicate timestamps;
- no missing hourly timestamps;
- no negative target values;
- 295 documented service-closure observations with zero rentals.

See [`docs/dataset_card.md`](docs/dataset_card.md) for provenance,
licensing, integrity information, variables, and limitations.

## Current status

Completed:

- reproducible repository foundation;
- Python package configuration;
- research-question documentation;
- dataset selection and acquisition;
- dataset provenance and licence documentation;
- raw-file integrity verification;
- initial structural data audit;
- automated dataset tests;
- reusable YAML data-configuration loader;
- reusable CSV data loader;
- explicit date, hour, and target parsing;
- construction of the canonical hourly timestamp;
- automated loader error-handling tests;
- structured time-series validation agent;
- duplicate, missing, irregular, and disordered timestamp checks;
- missing, negative, and zero target checks;
- documented-closure consistency checks;
- machine-readable JSON validation report;
- human-readable Markdown validation report;
- leakage-safe preprocessing module;
- chronological sorting without row deletion;
- structural-zero and closure preservation;
- processed-data and preprocessing-summary generation;
- descriptive time-series statistics;
- daily and weekly autocorrelation diagnostics;
- Augmented Dickey–Fuller stationarity test;
- rolling-mean and rolling-variability analysis;
- hourly demand profile;
- ACF and PACF figures;
- additive daily-seasonal decomposition;
- consistent baseline-model interface;
- mean and last-value forecasts;
- daily seasonal-naive forecasting;
- weekly seasonal-naive forecasting;
- validation of regular training timestamps;
- reproducible next-24-hour forecast preview;
- one-week chronological holdout evaluation;
- MAE, RMSE, sMAPE, and MASE implementation;
- leakage-safe MASE scaling from training data;
- baseline holdout comparison figure;
- 12-fold expanding-window validation;
- non-overlapping weekly test periods;
- fold-level and aggregate accuracy results;
- performance-variability reporting;
- fold-win counts and mean-MAE ranking;
- additive Holt-Winters forecasting;
- damped additive trend;
- daily seasonal period of 24 hours;
- residual and information-criterion diagnostics;
- reproducible next-24-hour Holt-Winters preview;
- transparent nonnegative count-forecast constraint;
- reporting of raw negative forecasts before clipping;
- 12-fold Holt-Winters evaluation;
- direct comparison with the weekly seasonal-naive benchmark;
- fold-level reporting of constrained negative forecasts;
- mean-error improvement and fold-win evidence;
- leakage-safe lag features;
- shifted rolling statistics;
- cyclical hour and weekday features;
- explicit current-target leakage tests;
- machine-readable feature summary;
- deterministic Gradient Boosting forecaster;
- recursive multi-step prediction;
- training and future-feature alignment;
- feature-importance diagnostics;
- nonnegative machine-learning forecasts;
- integration of Gradient Boosting with temporal evaluation;
- 12-fold recursive Gradient Boosting evaluation;
- comparison of six forecasting models on identical weekly folds;
- reporting of machine-learning raw negative forecasts;
- fold-level machine-learning accuracy results;
- machine-learning model-ranking figures;
- direct Gradient Boosting comparison with the weekly benchmark;
- evidence-based selection of the preferred forecasting model;
- reusable expanding-window residual collection;
- timestamp-level out-of-sample actual and forecast records;
- explicit residual and absolute-residual calculation;
- closure-status attachment to forecast residuals;
- preservation of fold-level raw-negative forecast counts;
- machine-readable residual-collection summary;
- verification that residual MAE reproduces evaluation MAE;
- robust median-and-MAD residual anomaly detection;
- modified-z-score anomaly thresholding;
- exclusion of known closures from anomaly calibration;
- separation of statistical and actionable alerts;
- positive-spike and negative-drop classification;
- moderate, high, and extreme anomaly severity;
- contextual weather and operating-day enrichment;
- consecutive-hour anomaly episode construction;
- rain-coincident demand-drop identification;
- recursive forecast-floor failure identification;
- ranked contextual anomaly evidence;
- anomaly timeline and threshold figures;
- daily anomaly-count visualization;
- reproducible human-readable anomaly report;
- transparent evidence-based model recommendation policy;
- minimum MAE-improvement decision gate;
- nondegradation RMSE decision gate;
- equal-fold comparability requirement;
- automatic preferred-model and fallback-model selection;
- model-complexity and raw-negative forecast reporting;
- explicit model-selection cautions;
- machine-readable model-recommendation JSON;
- human-readable model-recommendation report;
- operational forecasting and fallback policy.

Not yet completed:


- forecast-residual anomaly detection;
- anomaly scoring and explanation;
- final model-recommendation agent;
- end-to-end command-line pipeline;
- continuous-integration workflow;
- final reproducibility and portfolio report.


## Forecasting results

Six models were evaluated using 12 expanding-window folds. Each fold
forecast the next 168 hourly observations, producing 2,016
out-of-sample predictions per model.

| Rank | Model | Mean MAE | Mean RMSE | Mean sMAPE | Mean MASE | MAE fold wins |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Recursive Gradient Boosting | 404.72 | 544.54 | 71.57 | 1.508 | 3 |
| 2 | Weekly seasonal naive | 420.56 | 618.48 | 71.80 | 1.573 | 4 |
| 3 | Daily seasonal naive | 444.04 | 621.37 | 80.87 | 1.647 | 3 |
| 4 | Holt-Winters | 516.69 | 641.29 | 86.35 | 1.922 | 0 |
| 5 | Training mean | 523.50 | 632.75 | 80.16 | 1.953 | 1 |
| 6 | Last-value naive | 566.23 | 703.82 | 96.08 | 2.099 | 1 |

Recursive Gradient Boosting reduced mean MAE by 15.84 bikes, or 3.77%,
relative to the weekly seasonal-naive benchmark. It also achieved the
lowest mean RMSE.

The improvement is modest rather than universal. Gradient Boosting won
3 of the 12 folds, while the weekly seasonal-naive model won 4.
Gradient Boosting also produced 199 raw negative forecasts out of 2,016
predictions. These physically impossible values were recorded and then
constrained to zero.

Gradient Boosting is therefore the preferred accuracy model at the
current stage. Weekly seasonal naive remains the transparent,
naturally nonnegative fallback model.

## Residual anomaly results

Anomaly detection uses 2,016 out-of-sample residuals from the selected
recursive Gradient Boosting forecaster.

The robust reference distribution excludes 247 documented service
closures. A modified-z-score threshold of 3.5 identifies unusually large
forecast residuals.

| Measure | Result |
|---|---:|
| Out-of-sample residuals | 2,016 |
| Nonclosure calibration rows | 1,769 |
| Known closures | 247 |
| Statistical anomalies | 182 |
| Actionable candidate hours | 115 |
| Positive demand surprises | 87 |
| Negative demand surprises | 28 |
| Actionable candidate rate | 6.50% |
| Consecutive anomaly episodes | 37 |

The 115 actionable hours were grouped into consecutive episodes:

| Episode context | Episodes | Anomalous hours |
|---|---:|---:|
| Forecast-floor positive episode | 12 | 62 |
| Rain-coincident negative episode | 5 | 22 |
| Other residual episode | 20 | 31 |

The ten most concentrated dates contain 93 of the 115 actionable hours,
or 80.87%. This shows that alerts occur mainly in connected episodes
rather than as independent isolated points.

The 5 October 2018 rain episode contained 12 consecutive negative
anomaly hours. During these hours, observed demand totaled 396 rentals
against a forecast of 20,521, while accumulated rainfall was 31 mm.

The strongest positive episode occurred on 17 September 2018. The
forecast approached zero while actual demand remained high, indicating
a recursive forecast-floor and model-recovery failure.

These findings distinguish unusual observed demand from unusual
forecasting behavior. They are candidate alerts requiring contextual or
domain review because the dataset contains no verified anomaly labels.

## Model recommendation

The recommendation agent uses a transparent operational policy rather
than selecting a model from rank alone.

Recursive Gradient Boosting can replace the weekly seasonal-naive
benchmark only when:

1. its mean-MAE improvement is at least 2%;
2. its mean RMSE is no worse than the benchmark;
3. both models were evaluated over the same chronological folds.

All three checks passed:

| Decision check | Required | Observed | Result |
|---|---:|---:|---|
| Mean-MAE improvement | At least 2.00% | 3.77% | Pass |
| Mean-RMSE degradation | None | 11.95% improvement | Pass |
| Comparable validation | Same folds | 12 folds each | Pass |

The agent therefore recommends:

- preferred model: `gradient_boosting_recursive`;
- fallback model: `seasonal_naive_168`;
- decision status: `candidate_selected`.

The preferred model has high implementation complexity, while the
fallback has low complexity.

The recommendation records three cautions:

- Gradient Boosting has slightly greater fold-to-fold MAE variability;
- it wins fewer individual folds than the weekly benchmark;
- 9.87% of its raw predictions are negative before clipping.

The weekly seasonal-naive model remains the operational fallback because
it is simple, stable, naturally nonnegative, and does not suffer from
recursive forecast-floor failures.


## Repository structure

```text
configs/                 Project configuration
data/raw/                Original source data
data/interim/            Intermediate data
data/processed/          Model-ready data
docs/                    Research documentation
notebooks/               Exploratory notebooks
reports/                 Figures, metrics, and validation evidence
src/time_series_agent/   Python source code
tests/                   Automated tests
scripts/                 Reproducible execution scripts
```

## Command-line interface

Install the project in editable mode:

```powershell
python -m pip install -e ".[dev]"
```

View the available workflows:

```powershell
time-series-agent --help
```

Common commands include:

```powershell
time-series-agent forecast
time-series-agent anomalies
time-series-agent recommend
time-series-agent run-all --dry-run
```

The equivalent Python-module form is:

```powershell
python -m time_series_agent forecast
```

The CLI validates script availability, uses the active Python environment, executes dependent scripts in order, and stops immediately when a step fails.

See `docs/cli_guide.md` for complete usage instructions.

## Project status

The following major components are complete:

- reproducible data loading and validation;
- leakage-safe preprocessing and feature engineering;
- baseline, classical, and machine-learning forecasting;
- chronological holdout and expanding-window evaluation;
- residual-based anomaly detection and episode reporting;
- evidence-based model recommendation;
- unified command-line workflows;
- 151 automated tests with 90% package coverage.

Remaining work:

- continuous-integration workflow;
- final end-to-end project report and portfolio audit.

