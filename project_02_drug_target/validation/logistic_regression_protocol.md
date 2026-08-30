# Davis Logistic-Regression Protocol

## Purpose

This stage provides an interpretable, regularized logistic-regression
benchmark using the frozen Davis model inputs.

It follows the dummy baseline and is not intended to maximize benchmark
accuracy.

## Human Class-Imbalance Decision

The primary model uses:

`class_weight="balanced"`

An unweighted model, with `class_weight=None`, is reported as a pre-specified
sensitivity analysis.

The weighted result is the primary logistic-regression result. The unweighted
result is not selected or rejected according to test performance.

No synthetic oversampling, undersampling, or resampling is performed.

## Fixed Pipeline

Each variant uses:

`StandardScaler -> LogisticRegression`

The scaler is inside the fitted pipeline. Its mean and standard deviation are
computed from training data only, then applied to the held-out test data.

## Fixed Model Parameters

- solver: `liblinear`;
- regularization: L2;
- `l1_ratio=0.0`;
- `C=1.0`;
- `max_iter=1000`;
- `random_state=20260830`;
- fixed decision threshold: `0.50`.

The `C` value, solver, threshold, and feature set are not tuned on the outer
test set.

## Evaluation Scope

The initial report uses:

- label: `interaction_kd_le_1000_nM`;
- policy: `cold_drug`;
- 36 transparent feature columns;
- the shared binary evaluation module.

The report includes average precision, ROC-AUC, accuracy, precision, recall,
F1, confusion-matrix counts, intercept, convergence iterations, and one
coefficient for each standardized feature.

## Coefficient Interpretation

A positive standardized coefficient is associated with a higher fitted log-odds
of the positive benchmark label while other features are held constant in the
model.

A coefficient is not:

- a causal effect;
- a biochemical mechanism;
- evidence of direct drug binding;
- a clinical finding;
- a validated biomarker.

The descriptors are correlated, partially redundant, and intentionally simple.
Coefficient signs and ranks are descriptive model properties only.

## Overfitting and Leakage Safeguards

- All scaling occurs inside the training-fitted pipeline.
- The frozen outer test partition is never used to fit scaling or coefficients.
- No feature selection or hyperparameter search is performed.
- A convergence warning is treated as a failed run requiring investigation.
- The weighted and unweighted variants are reported together without choosing
  the better test result as the preferred model.