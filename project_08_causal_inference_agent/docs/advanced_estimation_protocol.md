# Advanced causal-effect estimation protocol

Phase 07 extends the transparent Phase 06 baselines with cross-fitting and a
forest-based heterogeneous-effect learner. Estimation remains downstream of
the Phase 05 identification gate.

## Supported estimators

### Cross-fitted AIPW

`cross_fitted_aipw` estimates the ATE. Stratified folds generate out-of-fold
logistic propensity predictions and separate linear outcome-regression
predictions. The resulting doubly robust scores provide the point estimate,
influence-score standard error, and normal-approximation confidence interval.

### Forest DR-learner

`dr_learner_forest` estimates CATE values. It first constructs cross-fitted
doubly robust pseudo-outcomes and then predicts them from the identified effect
modifiers with out-of-fold random-forest regressions. It reports one
cross-fitted effect prediction per observation, the mean prediction,
heterogeneity range, standard deviation, and averaged feature importances.

This method is deliberately named a forest DR-learner. It must not be described
as a generalized random forest or as EconML `CausalForestDML`.

## Mandatory refusal behavior

The module rejects:

1. every identification status other than `IDENTIFIED`;
2. unsupported method-estimand pairs;
3. CATE estimation without identified conditioning variables;
4. adjustment sets not returned by the identifier;
5. malformed cross-fitting, clipping, forest, or confidence settings;
6. missing, non-numeric, non-finite, or non-binary analysis data;
7. treatment groups too small for the requested stratified folds;
8. samples below the cross-fitting minimum; and
9. propensity clipping above the configured fraction.

## Uncertainty boundary

The ATE interval is an influence-score normal approximation and must later be
assessed through empirical coverage experiments. The forest DR-learner does
not report pointwise CATE confidence intervals: its `standard_error` and
`confidence_interval` are null by design. PEHE and heterogeneous-effect
calibration belong to the later evaluation phase.

Cross-fitting reduces bias from nuisance-model overfitting. It does not prove
exchangeability, positivity, consistency, correct intervention definition, or
correctness of the approved DAG. Every result remains human-reviewable.
