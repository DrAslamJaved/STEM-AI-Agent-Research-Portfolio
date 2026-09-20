# Causal refutation and stress-testing protocol

Phase 09 stress-tests an already identified and successfully estimated causal
effect. It never routes a rejected identification result around the existing
gate, and it never interprets a passing refuter as proof that the causal model
is true.

## Refuters

### Placebo treatment

The observed treatment is independently permuted for each simulation and the
same estimator is rerun. The empirical two-sided exceedance probability uses a
finite-simulation correction. A large probability triggers review because the
observed effect is not distinguishable from the placebo distribution under the
configured test.

### Random common cause

An independent Gaussian covariate is added to every identified adjustment set
and the estimator is rerun. Material absolute or relative movement triggers
review. The generated variable is a robustness perturbation, not a newly
asserted scientific cause.

### Stratified subset

Treatment-stratified subsets preserve at least two observations per treatment
group. Material instability across subsets triggers review.

## Deterministic decisions

Every refuter records successful and failed runs, the mean refuted estimate,
mean and maximum absolute shifts, relative shift, and—where defined—an
empirical p-value. Excess failed runs, unavailable refutation distributions,
or breached thresholds produce machine-readable `REVIEW` reasons.

## Interpretation boundary

A suite `PASS` means only that these configured perturbations did not detect
instability. It does not validate the DAG, exchangeability, positivity,
consistency, intervention definition, model specification, or absence of
hidden confounding. Dedicated hidden-confounding sensitivity analysis and
benchmark coverage experiments remain later phases.
