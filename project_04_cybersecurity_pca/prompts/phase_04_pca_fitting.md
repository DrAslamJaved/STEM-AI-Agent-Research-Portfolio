# Phase 04 Prompt: Normal-Only PCA Fitting

## Objective

Integrate the validated manual PCA implementation with the leakage-safe
standardized cybersecurity partitions.

## Scope

Implement only:

1. PCA fitting on the standardized normal-fit partition;
2. minimum-component selection using cumulative explained variance;
3. transformation of normal-fit, normal-calibration, and test partitions;
4. preservation of flow identifiers;
5. defensive validation and numerical evidence;
6. package exports, tests, documentation, and execution trace.

Reconstruction-error thresholding and anomaly classification are outside this
phase.

## Leakage contract

The PCA model must be fitted only on `normal_fit`.

The following must not influence PCA fitting, eigenvalues, eigenvectors,
component selection, or retained-component count:

- normal-calibration feature values;
- normal test feature values;
- attack feature values;
- anomaly labels;
- scenario labels.

## Component-selection rule

Given ordered explained-variance ratios \(r_1,\ldots,r_p\), select the minimum
integer \(k\) satisfying

\[
\sum_{j=1}^{k} r_j \geq \gamma,
\]

where the baseline target is

\[
\gamma=0.95.
\]

The target must be finite and satisfy

\[
0 < \gamma \leq 1.
\]

## Fitting procedure

1. Validate the standardized partitions.
2. Fit a full-component `ManualPCA` model on `normal_fit`.
3. calculate the complete eigenvalue and explained-variance evidence;
4. select the minimum retained-component count;
5. refit `ManualPCA` on the same normal-fit observations using the selected
   component count;
6. verify that full and selected fits produce consistent complete
   eigenvalues;
7. transform all three partitions with the selected model.

## Required API

Implement:

- `PCAFitResult`;
- `PCAScoreSplits`;
- `select_n_components`;
- `fit_normal_pca`;
- `transform_pca_splits`.

Export these symbols through `cyber_pca.__init__`.

## Validation requirements

Validate:

- standardized input type;
- nonempty partitions;
- exact feature columns and order;
- `flow_id` index name;
- nonmissing and unique flow identifiers;
- finite float64 feature values;
- at least two normal-fit observations;
- disjoint partition identifiers;
- valid explained-variance ratios;
- valid explained-variance target;
- fitted PCA model;
- expected model feature count.

## Mathematical evidence

Record:

- selected component count;
- target and achieved explained variance;
- complete eigenvalues;
- complete cumulative explained variance;
- covariance symmetry error;
- maximum eigenpair residual;
- eigenvector orthonormality error;
- PCA fitting mean;
- score shapes for all partitions.

## Testing requirements

Test:

- minimum component selection;
- normal-only fitting;
- invariance to calibration and test feature changes;
- deterministic repeated fitting;
- score shapes and flow identifiers;
- target and explained-variance validation;
- malformed standardized partitions;
- overlapping identifiers;
- invalid and unfitted models;
- incorrect model feature count;
- package exports.

Run the complete regression suite with at least 90% combined coverage.

## Evidence rules

Record only commands actually executed and their actual exit codes. Record
failures and corrections. Never fabricate tests, coverage, files, numerical
results, commits, or pushes.
