# Davis Training-Only Feature-Quality Protocol

## Purpose

This stage documents the quality of the transparent Davis feature
representation before tree-based modeling.

It is a descriptive audit, not a model-selection or biological-analysis
procedure.

## Frozen Scope

The audit uses only:

- the primary binary label definition:
  `interaction_kd_le_1000_nM`;
- the frozen `cold_drug` split policy;
- `ModelDataset.X_train`;
- the 36 transparent feature columns defined in
  `src.features.representations`.

The held-out test feature matrix and held-out test labels are not used to
calculate feature statistics, zero-variance features, or correlations.

Loading the fixed dataset validates artifact integrity and split membership,
but audit statistics are calculated from the training partition only.

## Checks

For every training feature, the audit verifies:

- numeric representation;
- no missing values;
- no non-finite values;
- unique-value count;
- minimum, maximum, mean, population standard deviation, and population
  variance.

The audit then reports:

- zero-variance descriptors;
- all absolute Pearson-correlation pairs at or above `0.95`;
- the feature columns retained after excluding zero-variance descriptors.

A high correlation is a descriptive screening result. It is not a hypothesis
test, causal result, feature importance score, or biological mechanism.

## Pre-Specified Feature-Quality Decisions

### Zero-Variance Features

A descriptor with zero variance in the relevant training data has no
discriminating information for that training fit.

Future model pipelines will include:

`VarianceThreshold(threshold=0.0)`

The selector must be fitted within the training pipeline, or within the
training fold during cross-validation. The raw feature table is not rewritten.

The existing logistic-regression holdout result remains an immutable record
and is not rerun or changed after this audit.

### Correlated Features

Highly correlated descriptors are retained and documented at this stage.

No automatic correlation pruning is performed because these simple
descriptors can be structurally related while still representing different
transparent properties. Correlation patterns will be considered when
describing model behavior, but they will not be used to select a model from
the outer test set.

No outcome-based feature selection is performed.

## Leakage and Overfitting Safeguards

- The frozen cold-drug test partition is excluded from audit calculations.
- The audit does not fit a predictive model.
- The audit does not use labels to rank or select features.
- Future preprocessing is fitted only on training data or training folds.
- Outer-test performance is never used to choose an audit threshold or a
  feature subset.

## Interpretation Boundary

Feature-quality findings describe the benchmark representation only.

They do not establish:

- biological activity;
- direct binding;
- biochemical mechanism;
- clinical relevance;
- statistical significance;
- causality.

Any biological follow-up requires independent evidence and appropriate
experimental validation.