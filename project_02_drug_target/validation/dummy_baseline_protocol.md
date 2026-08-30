# Davis Dummy-Baseline Protocol

## Purpose

The first classifier is a deliberately simple reference model:

`DummyClassifier(strategy="prior")`

It establishes the level of performance achievable without learning any
relationship between drug descriptors, target descriptors, and interaction
labels.

## Training Rule

The classifier is fit only on the training partition.

With strategy `prior`, the classifier:

- ignores all input feature values;
- learns the empirical class distribution in training labels;
- returns the same positive-class probability for every test pair;
- predicts the majority class at the fixed 0.50 decision threshold.

The recorded random state is `20260830`. The `prior` strategy is deterministic;
the seed is retained in the report for consistent experiment records.

## Evaluation Scope

The initial report uses:

- label: `interaction_kd_le_1000_nM`;
- policy: `cold_drug`;
- fixed outer holdout;
- fixed threshold: 0.50;
- shared evaluation module.

No hyperparameters are tuned. No resampling, scaling, feature selection, or
test-set-dependent threshold selection occurs.

## Expected Behavior

The cold-drug training positive rate is approximately 0.1851.

Therefore, every test pair receives a positive probability near 0.1851. Since
this is below 0.50, all test pairs are predicted negative.

This makes accuracy appear high because negatives are common, while recall and
F1 for positive interactions are zero. This is intentional evidence that
accuracy cannot be the headline metric for this imbalanced DTI task.

## Required Metrics

The report records:

- average precision, used as the project PR-AUC measure;
- ROC-AUC;
- accuracy;
- precision;
- recall;
- F1;
- true-negative, false-positive, false-negative, and true-positive counts.

## Interpretation Boundary

This baseline does not use biological or chemical information. It does not
predict binding mechanism, validate an interaction, provide statistical
evidence, or support causal claims.

It is only a lower-bound reference for later trained models.