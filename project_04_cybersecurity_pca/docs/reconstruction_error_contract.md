# Reconstruction Error and Threshold Contract

## 1. Objective

Use the retained normal-only PCA model to calculate observation-level
reconstruction errors, calibrate an anomaly threshold using only normal
calibration traffic, and generate label-free test predictions.

This phase implements detection logic. Precision, recall, F1, confusion
matrices, plots, and attack-specific evaluation belong to Phase 6.

## 2. Permitted inputs

Reconstruction errors may use:

- standardized model features;
- the retained `ManualPCA` model fitted during Phase 4;
- normal-fit, normal-calibration, and test feature partitions.

Threshold calibration may use only:

- reconstruction errors from the normal-calibration partition.

Threshold calibration must not use:

- normal-fit reconstruction errors;
- normal test reconstruction errors;
- attack reconstruction errors;
- test anomaly labels;
- scenario labels;
- evaluation metrics.

## 3. Reconstruction

For standardized observation \(z_i\), retained PCA components \(V_k\), and
PCA fitting mean \(\mu\), calculate:

\[
t_i=(z_i-\mu)V_k.
\]

The reconstruction is:

\[
\widehat z_i=t_iV_k^\top+\mu.
\]

All reconstruction calculations remain in standardized feature space.

## 4. Reconstruction error

For \(p\) model features, define observation-level mean squared reconstruction
error:

\[
e_i
=
\frac{1}{p}
\left\|
z_i-\widehat z_i
\right\|_2^2.
\]

The implementation must use float64 and return one finite, nonnegative error
for every observation.

Each error series must:

- preserve the original `flow_id` index;
- use index name `flow_id`;
- use series name `reconstruction_error`;
- contain no missing or infinite values;
- contain no negative values.

## 5. Error partitions

Define `ReconstructionErrorSplits` containing:

- `normal_fit`;
- `normal_calibration`;
- `test`.

Every field is a `pandas.Series` aligned with its corresponding standardized
feature partition.

Normal-fit errors are diagnostic only. They do not calibrate the anomaly
threshold.

## 6. Threshold calibration

For normal-calibration errors
\(e_1,\ldots,e_m\), use the configured quantile:

\[
q=0.99.
\]

Calculate:

\[
\tau
=
Q_q
\left(
e_1,\ldots,e_m
\right).
\]

Use NumPy's explicitly configured quantile method `linear` so the result does
not depend on an implicit library default.

The quantile must be a finite real number satisfying:

\[
0<q<1.
\]

Define `AnomalyThresholdResult` containing:

- `threshold`;
- `quantile`;
- `quantile_method`;
- `calibration_count`.

The threshold must be finite and nonnegative.

## 7. Prediction rule

For test reconstruction error \(e_i\), predict:

\[
\widehat y_i
=
\begin{cases}
1,& e_i>\tau,\\
0,& e_i\leq\tau.
\end{cases}
\]

The comparison is strictly greater than.

An error exactly equal to the threshold is therefore classified as normal.

Predictions must:

- preserve the test `flow_id` index;
- use series name `predicted_anomaly`;
- use integer values 0 and 1;
- contain no missing values.

## 8. Required API

Implement in `src/cyber_pca/detector.py`:

- `ReconstructionErrorSplits`;
- `AnomalyThresholdResult`;
- `compute_reconstruction_errors`;
- `calibrate_anomaly_threshold`;
- `predict_anomalies`.

Export these symbols through `cyber_pca.__init__`.

## 9. Leakage invariants

Automated tests must prove:

1. the PCA model remains fitted only on normal-fit observations;
2. threshold calibration reads only `normal_calibration`;
3. changing normal-fit errors does not change the threshold;
4. changing test errors does not change the threshold;
5. reordering test observations does not change the threshold;
6. test labels and scenarios are not accepted by the calibration API;
7. repeated calibration produces the same threshold.

## 10. Numerical validation

Tests must verify:

- manual reconstruction-error formula agreement;
- float64 output;
- nonnegative finite errors;
- exact partition shapes;
- identifier preservation;
- deterministic repeated execution;
- full-component reconstruction errors approach zero;
- reduced-component errors are generally positive;
- explicit agreement with `numpy.quantile` using `method="linear"`;
- strict threshold comparison;
- threshold-tie classification as normal.

## 11. Why large reconstruction error can indicate an anomaly

PCA fitted on representative normal traffic learns the dominant linear
correlation structure of normal observations.

An observation with a large component outside that retained normal subspace
cannot be reconstructed accurately. A large residual therefore indicates that
the observation is inconsistent with the learned normal linear structure.

This is evidence of statistical abnormality, not proof of malicious activity.

## 12. When the reasoning can fail

Large reconstruction error may be misleading when:

- normal structure is nonlinear;
- features are poorly scaled;
- legitimate observations are rare but valid;
- high-dimensional noise distorts the learned subspace;
- the threshold is inappropriate;
- training or calibration data are contaminated;
- normal behaviour changes through concept drift;
- attacks lie inside retained high-variance directions.

These limitations must remain visible in final reporting.

## 13. Phase boundary

Phase 5 ends after reconstruction errors, threshold calibration, and test
predictions are validated.

Phase 6 will use hidden test labels to calculate precision, recall, F1,
confusion-matrix counts, false positives, false negatives, and attack-specific
results.
