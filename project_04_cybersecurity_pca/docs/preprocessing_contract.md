# Leakage-Safe Preprocessing Contract

## 1. Objective

Define a deterministic preprocessing workflow that prevents attack labels,
calibration observations, and test observations from influencing PCA training.

The preprocessing stage must produce:

- a normal-only PCA fitting split;
- a normal-only threshold-calibration split;
- a labelled evaluation split;
- standardized feature matrices;
- fitted scaling statistics derived only from the normal fitting split.

## 2. Split roles

| Split | Permitted observations | Purpose |
|---|---|---|
| Fitting | Normal only | Fit feature scaling and PCA |
| Calibration | Normal only | Select the reconstruction-error threshold |
| Test | Held-out normal plus every attack observation | Final evaluation only |

Attack observations must never enter the fitting or calibration splits.

## 3. Default split proportions

Normal observations use the following default allocation:

| Normal split | Fraction |
|---|---:|
| Fitting | 0.60 |
| Calibration | 0.20 |
| Test | 0.20 |

For the default synthetic dataset, this produces:

| Partition | Observations |
|---|---:|
| Normal fitting | 2,400 |
| Normal calibration | 800 |
| Normal test | 800 |
| Attack test | 1,000 |
| Complete test split | 1,800 |

All 5,000 input observations must occur in exactly one partition.

## 4. General count rule

For \(n_{\mathrm{normal}}\) normal observations:

\[
n_{\mathrm{fit}}
=
\left\lfloor
0.60n_{\mathrm{normal}}
\right\rfloor,
\]

\[
n_{\mathrm{calibration}}
=
\left\lfloor
0.20n_{\mathrm{normal}}
\right\rfloor.
\]

All remaining normal observations belong to the normal test subset.

The implementation must reject split settings that produce an empty fitting,
calibration, or normal-test partition.

## 5. Deterministic splitting requirements

The splitting function must:

1. accept a nonnegative integer random seed;
2. use `numpy.random.default_rng`;
3. validate the complete input schema;
4. reject missing or duplicated flow identifiers;
5. reject invalid anomaly labels;
6. separate normal and attack observations before splitting;
7. permute normal observations deterministically;
8. allocate normal observations according to the configured fractions;
9. place all attack observations in the test split;
10. deterministically shuffle the complete test split;
11. preserve the original `flow_id` values;
12. return new DataFrames without mutating the input.

Identical input, configuration, and seed must produce identical partitions.

## 6. Partition invariants

The following conditions are mandatory:

- fitting labels are all 0;
- calibration labels are all 0;
- the test split contains labels 0 and 1;
- no flow identifier occurs in more than one split;
- no flow identifier is lost;
- the union of split identifiers equals the input identifiers;
- feature order remains unchanged;
- scenario labels remain aligned with their observations.

## 7. Standardization mathematics

For fitting observation \(i\) and feature \(j\), calculate:

\[
z_{ij}
=
\frac{x_{ij}-\mu_j}{\sigma_j}.
\]

The feature mean is:

\[
\mu_j
=
\frac{1}{n_{\mathrm{fit}}}
\sum_{i=1}^{n_{\mathrm{fit}}}x_{ij}.
\]

The scaler uses the population variance convention implemented by
`sklearn.preprocessing.StandardScaler`:

\[
\sigma_j^2
=
\frac{1}{n_{\mathrm{fit}}}
\sum_{i=1}^{n_{\mathrm{fit}}}
(x_{ij}-\mu_j)^2.
\]

PCA will later calculate covariance using the \(n-1\) sample denominator. The
scaler variance convention and PCA covariance convention must therefore be
documented separately.

## 8. Leakage-safe scaling requirements

The standardizer must:

1. fit only on normal fitting features;
2. exclude `flow_id`, `is_anomaly`, and `scenario`;
3. retain the documented ten-feature order;
4. transform fitting, calibration, and test features using the same fitted
   means and scales;
5. produce float64 values;
6. reject missing, infinite, or incorrectly ordered features;
7. reject zero-variance fitting features;
8. preserve flow identifiers for alignment;
9. support approximate inverse transformation.

The fitting feature matrix should have means approximately zero and population
standard deviations approximately one.

Calibration and test matrices must not be forced to have zero means or unit
standard deviations.

## 9. Leakage tests

Automated tests must demonstrate that:

- attack rows cannot enter fitting or calibration;
- changing test-feature values does not change fitted scaler statistics;
- changing calibration-feature values does not change fitted scaler statistics;
- test labels do not influence fitting;
- standardization uses fitting means and scales for every split;
- inverse transformation approximately reproduces the original features.

A deliberately extreme test observation should remain extreme after
transformation rather than changing the fitted scaler.

## 10. Planned data structures

The implementation should provide an immutable raw-split container holding:

- normal fitting observations;
- normal calibration observations;
- labelled test observations.

A second immutable container should hold:

- standardized fitting features;
- standardized calibration features;
- standardized test features;
- the fitted scaler;
- aligned flow identifiers.

## 11. Scope limitation

Phase 3 implements splitting and standardization only.

It does not fit PCA, select principal components, calibrate an anomaly
threshold, predict anomalies, or calculate final classification metrics.
