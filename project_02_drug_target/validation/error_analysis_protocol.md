# Davis Selected-Model Error-Analysis Protocol

## Purpose

This stage investigates where the selected binary DTI model succeeds and fails
without using the frozen outer cold-drug holdout for further development.

It is an error-characterization exercise, not a further model-selection or
hyperparameter-tuning procedure.

## Selection Input

The selected model is determined automatically from:

`reports/davis_inner_cold_drug_cv.json`

Selection uses the pre-specified unweighted mean inner-fold average precision.

For the current Davis experiment, the expected selected candidate is:

- model: `random_forest_balanced`;
- selection metric: mean five-fold inner cold-drug average precision;
- runner-up comparison: histogram gradient boosting.

The small difference between candidates is an operational selection result. It
does not establish statistical superiority.

## Data Scope and Leakage Guard

The analysis reads only:

- the frozen inner-CV summary;
- the inner out-of-fold prediction file;
- predictions generated from the 54 outer-training drugs.

Each out-of-fold prediction is made by a model that was not trained on that
drug. The outer 14-drug test partition is not read, ranked, tuned against, or
used to modify the selected model.

The program stops if the inner-CV report does not explicitly state:

```text
outer_test_partition_used = false
cv_scope = frozen_outer_training_partition_only