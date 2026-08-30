# Normal-Only PCA Fitting Contract

## 1. Objective

Fit the validated manual PCA implementation using only standardized normal
fitting observations and select the smallest number of principal components
that satisfies the configured explained-variance target.

Phase 4 connects:

- deterministic cybersecurity data;
- leakage-safe preprocessing;
- covariance-based manual PCA;
- eigenvalue analysis;
- explained-variance component selection;
- principal-component scores.

## 2. Permitted fitting data

PCA may use only:

- the ten standardized features;
- from the normal fitting partition.

PCA must not use:

- calibration observations;
- test observations;
- attack observations;
- `flow_id`;
- `is_anomaly`;
- `scenario`;
- anomaly thresholds;
- evaluation metrics.

## 3. PCA covariance

For the standardized normal fitting matrix
\(Z_{\mathrm{fit}}\in\mathbb{R}^{n\times p}\), PCA centers the matrix using its
fitting-column means:

\[
Z_c
=
Z_{\mathrm{fit}}
-
\overline Z_{\mathrm{fit}}.
\]

The covariance matrix is:

\[
C
=
\frac{1}{n-1}
Z_c^\top Z_c.
\]

The eigensystem satisfies:

\[
Cv_j
=
\lambda_jv_j.
\]

Eigenvalues and corresponding eigenvectors must be ordered from largest to
smallest eigenvalue.

## 4. Explained variance

For eigenvalue \(\lambda_j\), the explained-variance ratio is:

\[
r_j
=
\frac{\lambda_j}
{\sum_{\ell=1}^{p}\lambda_\ell}.
\]

The cumulative explained variance for the first \(k\) components is:

\[
R_k
=
\sum_{j=1}^{k}r_j.
\]

## 5. Component-selection rule

The default explained-variance target is:

\[
\gamma=0.95.
\]

Select:

\[
k
=
\min
\left\{
m:
R_m\geq\gamma
\right\}.
\]

The selected component count must therefore satisfy:

- achieved cumulative variance is at least the target;
- if \(k>1\), cumulative variance using \(k-1\) components is below the target;
- \(1\leq k\leq p\).

The implementation must reject nonfinite targets and targets outside
\(0<\gamma\leq1\).

## 6. Two-stage fitting procedure

The integration layer should:

1. fit a full-component `ManualPCA` model on standardized normal fitting data;
2. calculate full cumulative explained variance;
3. select the minimum valid component count;
4. fit the retained-component `ManualPCA` model using the same normal fitting
   data;
5. verify that both fits have consistent full eigensystems.

The two-stage procedure must be deterministic.

## 7. Required result object

The PCA fitting result should preserve:

- the retained-component PCA model;
- selected component count;
- requested explained-variance target;
- achieved explained variance;
- full eigenvalues;
- full explained-variance ratios;
- full cumulative explained variance.

The result object must not contain anomaly labels or classification metrics.

## 8. Principal-component scores

For retained eigenvectors \(V_k\), calculate:

\[
T
=
Z_cV_k.
\]

The same retained model must transform:

- normal fitting features;
- normal calibration features;
- test features.

The fitting, calibration, and test score matrices must preserve their original
flow identifiers.

Their shapes must be:

- fitting: \(n_{\mathrm{fit}}\times k\);
- calibration: \(n_{\mathrm{calibration}}\times k\);
- test: \(n_{\mathrm{test}}\times k\).

## 9. Leakage tests

Automated tests must prove that:

- changing calibration values does not change the fitted PCA model;
- changing test values does not change the fitted PCA model;
- changing test labels does not change the fitted PCA model;
- only fitting-feature values affect covariance and eigenvectors;
- metadata columns never enter PCA.

Comparison should use eigenvalues and projection matrices rather than raw
eigenvector signs.

## 10. Mathematical validation

Tests must verify:

- covariance symmetry;
- descending eigenvalues;
- nonnegative eigenvalues within tolerance;
- eigenvector orthonormality;
- eigenpair consistency;
- cumulative explained variance monotonicity;
- minimum-component selection;
- achieved explained variance;
- deterministic repeated fitting;
- score dimensions and identifier alignment.

## 11. Scope limitation

Phase 4 does not calibrate an anomaly threshold or calculate precision, recall,
F1, false positives, or false negatives.

Reconstruction errors may be inspected for mathematical behavior, but anomaly
classification belongs to the next phase.
