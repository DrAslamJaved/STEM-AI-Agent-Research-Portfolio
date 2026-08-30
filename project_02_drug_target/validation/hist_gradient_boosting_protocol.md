# Davis Histogram Gradient-Boosting Protocol

## Purpose

This stage supplies the stronger machine-learning comparison required by the
project, using scikit-learn's histogram gradient-boosting classifier.

It is included because the Davis training partition contains 23,868 interaction
pairs and may contain nonlinear relationships not represented by the logistic
baseline or bagged Random Forest.

It is not a hyperparameter search and is not intended to maximize holdout
performance.

## Frozen Evaluation Scope

- primary label: `interaction_kd_le_1000_nM`;
- policy: `cold_drug`;
- frozen train/test assignment;
- decision threshold: `0.50`;
- shared binary evaluation module.

The outer test partition is used only after fitting the complete training
pipeline.

## Fixed Pipeline

```text
VarianceThreshold(threshold=0.0) -> HistGradientBoostingClassifier