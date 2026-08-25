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
4. Construct mean, naïve, and seasonal-naïve baselines.
5. Train classical forecasting models.
6. Train one lag-based machine-learning model.
7. Evaluate models using chronological validation.
8. Detect anomalies from forecast residuals.
9. recommend a model using quantitative evidence.
10. Generate reproducible reports.

## Current status

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
- processed-data and preprocessing-summary generation.

Not yet completed:


- exploratory analysis;
- forecasting models;
- anomaly detection;
- model recommendation

No forecasting models have yet been trained, and no forecast-accuracy
results are currently reported.

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