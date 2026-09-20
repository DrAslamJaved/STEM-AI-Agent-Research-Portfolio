# Diagnostic policy

Identification is conceptual; empirical estimability is data-dependent. Phase
08 therefore audits covariate balance, propensity overlap, causal weights, and
effective sample size before an estimate may be presented as reliable.

## Decision rule

The report emits `PASS` only when every configured gate passes. It emits
`REVIEW` when any of the following occurs:

1. the largest absolute weighted standardized mean difference exceeds its
   threshold;
2. the fraction inside the declared propensity-overlap interval is too small;
3. the effective-sample-size fraction is too small;
4. at least one causal weight exceeds the configured maximum; or
5. the fraction of propensities requiring numerical clipping exceeds its
   threshold.

The report preserves both unweighted and weighted covariate-level standardized
mean differences. A zero pooled variance with unequal group means is treated as
infinite imbalance, not perfect balance.

## Supported targets

- ATE weights: `T/e(X) + (1-T)/(1-e(X))`;
- ATT weights: `T + (1-T)e(X)/(1-e(X))`.

Propensity values must be finite and inside `[0, 1]`. Numerical clipping is
reported explicitly and can itself trigger review.

## Interpretation boundary

A `PASS` result does not prove exchangeability, positivity in the target
population, consistency, correct intervention definition, or correctness of
the approved DAG. A `REVIEW` result is a deterministic stop-and-investigate
signal. Trimming, alternative weighting, or population restriction requires a
new analysis and an explicit changed-target-population warning.
