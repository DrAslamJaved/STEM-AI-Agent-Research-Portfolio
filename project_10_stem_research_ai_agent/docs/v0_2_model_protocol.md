# v0.2 Reproducible DTI Baseline Protocol

## Feature set

The initial v0.2 models use fixed, interpretable features only: SMILES length
and token frequencies, plus protein length and amino-acid composition. These
features are deliberately modest. They provide auditable reference results, not
a claim of state-of-the-art molecular representation learning.

## Models

1. Training-prevalence score: a no-feature reference.
2. Class-balanced logistic regression with training-only standardisation.
3. Class-balanced random forest with 200 trees and a fixed random seed.

## Metrics and interpretation

Every model reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and Brier
score. The positive class represents pKd >= 7.0. Given the approximately 8%
positive prevalence, PR-AUC and calibration must be interpreted alongside
ROC-AUC. Results are reported separately for pair-random, cold-drug, and
cold-target conditions; pair-random results are not evidence of cold-start
generalization.

## Reproducibility

The initial run uses split seed `20260915`, test fraction 0.2, and a
single-threaded random forest. Any parameter change defines a distinct run.
