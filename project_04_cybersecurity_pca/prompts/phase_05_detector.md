# Phase 05 Implementation Prompt

## Project

Agentic Cybersecurity Anomaly Detection Using PCA and Eigenvalue Analysis

## Scope

Implement reconstruction-error calculation, normal-only threshold calibration,
and binary anomaly prediction for the deterministic synthetic cybersecurity
workflow.

Do not implement evaluation metrics, plots, UNSW-NB15 processing, or agent
reasoning evaluation during this phase.

## Existing validated inputs

The implementation must reuse:

- leakage-safe standardized partitions;
- PCA fitted only on normal fitting observations;
- the minimum component count reaching 95% explained variance;
- the existing ManualPCA reconstruction methods.

The expected standardized partitions are:

- 2,400 normal fitting observations;
- 800 normal calibration observations;
- 1,800 test observations.

## Reconstruction-error contract

For standardized observation \(z_i\) and its PCA reconstruction
\(\widehat z_i\), calculate:

\[
e_i=\frac{1}{p}\lVert z_i-\widehat z_i\rVert_2^2.
\]

Calculate errors independently for normal fitting, normal calibration, and test
partitions.

Every output must:

- be a float64 pandas Series;
- be named `reconstruction_error`;
- preserve the original `flow_id` index;
- contain finite, nonnegative values;
- avoid mutation of its input.

## Threshold contract

Use only normal calibration reconstruction errors.

Calculate:

\[
\tau=Q_{0.99}(e_1,e_2,\ldots,e_m)
\]

with NumPy quantile method `linear`.

The threshold must not use:

- fitting errors;
- test errors;
- anomaly labels;
- scenario labels;
- test-set statistics.

## Prediction contract

Predict an anomaly using the strict rule:

\[
\widehat y_i =
\begin{cases}
1, & e_i>\tau,\\
0, & e_i\leq\tau.
\end{cases}
\]

An error exactly equal to the threshold must be classified as normal.

Predictions must:

- be an int8 pandas Series;
- be named `is_anomaly`;
- preserve the test `flow_id` index;
- contain only zero and one;
- be deterministic.

## Required public API

Implement and export:

- `ReconstructionErrorSplits`;
- `AnomalyThresholdResult`;
- `compute_reconstruction_errors`;
- `calibrate_anomaly_threshold`;
- `predict_anomalies`.

## Required validation

Test:

- reconstruction-error agreement with the explicit matrix formula;
- agreement with `ManualPCA.reconstruction_error`;
- threshold agreement with explicit `numpy.quantile`;
- calibration-only thresholding;
- strict greater-than prediction;
- threshold-tie behaviour;
- deterministic execution;
- index, dtype, shape, and naming contracts;
- invalid types and empty inputs;
- missing and duplicate identifiers;
- nonnumeric, nonfinite, and negative values;
- invalid quantiles and unsupported quantile methods;
- full-component reconstruction behaviour;
- complete default synthetic integration;
- package-level public exports.

## Evidence requirements

Run the complete suite with:

- explicit `--cov=cyber_pca`;
- a 90% minimum combined coverage gate;
- JUnit XML output;
- coverage XML output;
- deprecation warnings treated as errors.

Record only commands actually executed, genuine failures, corrections, and
verified outcomes.

Do not report precision, recall, F1, confusion matrices, false positives, or
false negatives during Phase 5.
