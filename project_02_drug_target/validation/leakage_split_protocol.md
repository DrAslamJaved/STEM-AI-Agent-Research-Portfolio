# Davis Leakage-Aware Split Protocol

## Purpose

Drug-target interaction performance must be interpreted relative to the kind of
generalization being tested. A random pair split can share drugs and targets
between training and test data, which can produce an optimistic estimate for
claims about previously unseen entities.

## Fixed Outer Test Splits

For `cold_drug`, the outer test set is fold 4 (zero-indexed) from
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260830)`.

This fold was selected before model fitting using only the smallest absolute
difference between training and test positive-label prevalence among the five
deterministic candidate folds. It contains 14 unseen drugs and has prevalence
18.506% in training and 18.487% in testing.

This label-balance choice is documented as an engineering safeguard, not a
model-performance optimization. The model-evaluation phase will additionally
report all five drug-group folds.

## Leakage Safeguards

- Split assignments are created before feature fitting and model training.
- The same fixed assignments will be used for both binary-label variants.
- Feature vectorizers, scalers, imputation, feature selection, resampling, and
  hyperparameter selection must use training data only.
- Inner cross-validation must match the outer policy: grouped by drug for
  `cold_drug` and grouped by target for `cold_target`.
- Outer test data must not be used to select models, thresholds, or features.
- Entity overlap counts will be reported explicitly for every split.

## Interpretation Boundary

A lower score under cold-drug or cold-target evaluation does not invalidate a
random-pair model; it measures a harder and different prediction task.
Predictive performance does not establish biological mechanism or causality.