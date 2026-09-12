# Phase 07 — Comparison-Baseline Contract

## Fuzzy Dice

\[
D(U,V)=\frac{2\sum_i\min(\mu_U(x_i),\mu_V(x_i))}{\sum_i\mu_U(x_i)+\sum_i\mu_V(x_i)}.
\]

When the denominator is zero, the implementation returns `0.0`. Consequently,
the baseline does not satisfy reflexivity at the all-zero vector under this
convention. This is a declared comparison convention, not a defect.

## Fuzzy cosine

\[
C(U,V)=\frac{\sum_i\mu_U(x_i)\mu_V(x_i)}
{\sqrt{\sum_i\mu_U(x_i)^2}\sqrt{\sum_i\mu_V(x_i)^2}}.
\]

If either norm is zero, the implementation returns `0.0`; accordingly, it is
not reflexive at the all-zero vector under this convention.

## Common contract

Both baselines accept only finite, non-empty, equal-length vectors in `[0,1]`.
Their known values, symmetry, range on fixed-seed random nonzero vectors, and
zero conventions are tested. These tests validate code and conventions only;
they do not establish theorems or compare empirical performance.
