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

Completed:

- initial repository structure;
- Python package configuration;
- research-question document;
- dataset-card template;
- initial package test.

Not yet completed:

- dataset selection;
- data loading and validation;
- exploratory analysis;
- forecasting models;
- anomaly detection;
- model recommendation.

No empirical results are currently reported.

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