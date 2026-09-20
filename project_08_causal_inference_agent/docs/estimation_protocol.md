# Causal effect estimation baseline protocol

Phase 06 estimates an effect only after Phase 05 returns `IDENTIFIED`. Rejected,
unsupported, or under-specified identification results cannot reach an
estimator.

## Baseline estimators

- `difference_in_means`: allowed only when the identified adjustment set is
  empty;
- `g_computation`: additive linear outcome regression for ATE or ATT;
- `ipw`: logistic-propensity inverse weighting for ATE or ATT; and
- `aipw`: plug-in augmented inverse weighting for ATE.

All estimators report a point estimate, approximate standard error, normal
confidence interval, sample size, adjustment set, diagnostics, and explicit
assumption warning. These are transparent research baselines, not final
production estimators.

## Mandatory guards

The estimation gate rejects:

1. any identification state other than `IDENTIFIED`;
2. CATE and methods outside the Phase 06 contract;
3. adjustment sets not returned by the identifier;
4. naive differences when adjustment is required;
5. missing, non-numeric, non-finite, or non-binary data;
6. treatment groups with fewer than two observations;
7. rank-deficient outcome-regression designs; and
8. propensity overlap failures above the configured clipping fraction.

## Inference boundary

The reported intervals are baseline normal approximations. Outcome and
propensity models are fitted on the analysis sample without cross-fitting.
Consequently, interval coverage must be evaluated empirically in a later phase
and the output always retains `requires_human_review = true`.

No estimate proves exchangeability, positivity, consistency, intervention
validity, or correctness of the approved DAG. Diagnostics can reveal problems
but cannot validate untestable assumptions.
