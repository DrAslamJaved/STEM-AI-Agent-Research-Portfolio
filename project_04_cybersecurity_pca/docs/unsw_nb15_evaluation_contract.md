# UNSW-NB15 Evaluation and Reporting Contract

## Purpose

Phase 8 evaluates the frozen PCA reconstruction-error detector on the official
curated UNSW-NB15 testing partition. The phase reports an untuned observed
baseline and does not optimize the detector using official test outcomes.

## Frozen inputs

Phase 8 consumes the deterministic Phase 7 outputs:

- 42,000 normal fitting observations;
- 14,000 normal calibration observations;
- 82,332 official testing observations;
- 64 standardized model features;
- the normal-fit-only categorical encoder;
- the normal-fit-only standardizer;
- partition-local identifiers represented by the composite record key
  `(source_partition, id)`.

Training attack observations remain excluded from model development.

## Leakage boundary

The required execution order is:

1. fit the encoder and standardizer using normal fitting observations only;
2. fit PCA using normal fitting observations only;
3. select the minimum component count satisfying the 0.95 explained-variance
   target using normal fitting observations only;
4. compute reconstruction errors for fitting, calibration, and official test
   feature matrices;
5. calibrate the 0.99 quantile threshold using normal calibration errors only;
6. freeze official test predictions using the strict greater-than comparison;
7. access official test labels and attack categories only after predictions
   are frozen;
8. align predictions and hidden labels by `(source_partition, id)`;
9. calculate metrics once without post-evaluation tuning.

Test labels, attack categories, evaluation metrics, and official test errors
must not alter preprocessing, PCA, component selection, or threshold
calibration.

## PCA and threshold contract

The PCA implementation uses covariance eigenvalue analysis in standardized
feature space. The selected component count is the smallest count whose
cumulative explained-variance ratio is at least 0.95.

The anomaly threshold is the linear 0.99 quantile of reconstruction errors from
normal calibration observations. An observation is predicted anomalous only
when its reconstruction error is strictly greater than the frozen threshold.

## Evaluation alignment

Official curated identifiers are partition-local. The testing key is therefore
constructed as `unsw_testing:{id}` and named `flow_id`.

The reconstruction-error, prediction, and hidden-label identifier sets must be
identical. Positional alignment without identifier verification is forbidden.

The aligned evaluation table contains exactly:

- `true_anomaly`;
- `predicted_anomaly`;
- `scenario`;
- `reconstruction_error`.

## Binary metrics

The positive label is attack (`1`) and the negative label is normal (`0`).
Reported binary measures are:

- confusion matrix in label order `(0, 1)`;
- precision;
- recall;
- F1;
- accuracy;
- false-positive rate;
- false-negative rate;
- class support and prediction counts.

Zero denominators use the established project zero-division policy.

## Attack-category reporting

Results are reported in the official fixed order:

1. Normal;
2. Analysis;
3. Backdoor;
4. DoS;
5. Exploits;
6. Fuzzers;
7. Generic;
8. Reconnaissance;
9. Shellcode;
10. Worms.

For each category, the report records support, predicted-normal count,
predicted-anomaly count, predicted-anomaly rate, and reconstruction-error
summary statistics.

## Observed baseline

The frozen official result is:

- selected principal components: 34;
- achieved explained variance: `0.9521414327676875`;
- frozen threshold: `0.4923769885740442`;
- predicted normal: 78,951;
- predicted anomaly: 3,381;
- true negatives: 35,974;
- false positives: 1,026;
- false negatives: 42,977;
- true positives: 2,355;
- precision: `0.6965394853593612`;
- recall: `0.05195005735462808`;
- F1: `0.09668876891178946`;
- accuracy: `0.4655419520963902`;
- false-positive rate: `0.02772972972972973`;
- false-negative rate: `0.9480499426453719`.

These values expose weak attack recall under the untuned reconstruction-error
baseline. Shellcode and Worms have zero detected observations. These outcomes
must be reported rather than used to tune Phase 8.

## Permanent artifacts

Phase 8 writes:

- `results/unsw_nb15_evaluation.json`;
- `results/unsw_nb15_predictions.csv`;
- `reports/tables/unsw_nb15_metrics.csv`;
- `reports/tables/unsw_nb15_attack_category_metrics.csv`;
- `reports/figures/unsw_nb15_confusion_matrix.png`;
- `reports/figures/unsw_nb15_reconstruction_errors.png`;
- `reports/figures/unsw_nb15_scree_plot.png`;
- `reports/figures/unsw_nb15_attack_category_rates.png`.

A complete command-line regeneration must reproduce all eight artifacts
byte-for-byte.

## Interpretation

This result is evidence about one fixed PCA baseline, not a claim of
operational cybersecurity readiness. The low recall shows that variance-based
normal-subspace reconstruction alone does not separate most official attacks
at the calibration-only 0.99 threshold.

Future phases may compare models or add agent reasoning, but Phase 8 artifacts
must remain frozen and must not be retrospectively optimized.
