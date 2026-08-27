# Phase 1 Implementation Specification

## Scope

Implement and validate only:

1. the Python package foundation;
2. covariance-based manual PCA;
3. symmetric eigenvalue decomposition;
4. transformation and inverse transformation;
5. reconstruction and observation-level error;
6. independent mathematical validation;
7. the `validate-math` CLI;
8. genuine test and coverage evidence.

Do not download UNSW-NB15 or implement the final anomaly detector during
Phase 1.

## Mathematical contract

For centered observations:

\[
C=\frac{1}{n-1}X_c^TX_c.
\]

Use `numpy.linalg.eigh`, sort eigenpairs in descending eigenvalue order, and
calculate:

\[
T=X_cV_k,
\]

\[
\widehat X=TV_k^T+\mu,
\]

and

\[
e_i=\operatorname{mean}
\left[(x_i-\widehat x_i)^2\right].
\]

## Validation contract

Verify:

- finite two-dimensional input;
- covariance symmetry;
- descending eigenvalues;
- eigenpair consistency;
- eigenvector orthonormality;
- nonnegative eigenvalues within numerical tolerance;
- explained-variance consistency;
- full-component reconstruction;
- nonincreasing aggregate reconstruction error;
- scikit-learn principal-subspace agreement;
- near-collinear numerical stability;
- deterministic execution.

## Evidence policy

Record only commands actually executed and results actually observed.

Anything not executed must be marked:

```text
TO BE EXECUTED/VERIFIED