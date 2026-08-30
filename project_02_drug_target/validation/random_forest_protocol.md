# Davis Random Forest Protocol

## Purpose

This stage provides the required tree-based DTI model using the frozen Davis
cold-drug evaluation design.

It tests whether a conservative nonlinear ensemble improves prediction beyond
the dummy and logistic-regression baselines. It is not a hyperparameter search
and is not designed to maximize holdout accuracy.

## Frozen Evaluation Scope

- primary label: `interaction_kd_le_1000_nM`;
- policy: `cold_drug`;
- training pairs: 23,868;
- held-out test pairs: 6,188;
- decision threshold: `0.50`;
- evaluation: shared binary evaluation module.

The outer test partition is used once for final evaluation only.

## Fixed Pipeline

```text
VarianceThreshold(threshold=0.0) -> RandomForestClassifier