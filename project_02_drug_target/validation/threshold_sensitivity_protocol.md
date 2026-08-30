Davis Binary-Threshold Sensitivity Protocol

Purpose

This pre-specified sensitivity analysis asks whether descriptive inner
cold-drug performance changes when a positive Davis interaction is defined as
Kd less than or equal to 100 nM rather than Kd less than or equal to 1,000 nM.

It evaluates label-definition robustness. It does not choose a new model,
tune hyperparameters, tune a probability threshold, or revise the primary
model-selection decision.

Frozen Label Definitions

Variant

Label column

Positive definition

pKd definition

Role

Primary

interaction_kd_le_1000_nM

Kd <= 1,000 nM

pKd >= 6

Main DTI task

Sensitivity

interaction_kd_le_100_nM

Kd <= 100 nM

pKd >= 7

Stricter affinity task

The thresholds were specified before this stage. A change in threshold changes
prevalence and the prediction target, so performance values are not directly
interchangeable across variants.

Leakage-Safe Data Scope

The analysis is restricted to the 23,868 pairs from the 54 drugs in the frozen
outer cold-drug training partition. It obtains membership of those rows and
their five validation folds from the existing primary inner-CV OOF file:

data/interim/davis_inner_cold_drug_oof_predictions.csv

Only structural columns are read from that file:

model_id;

fold_index;

observed_pair_index;

drug_id;

target_id.

The source rows are taken from the dummy_prior candidate because it contains
one structural OOF assignment for every outer-training pair. Its prior labels
and probability values are deliberately not read.

The script rejects the run unless the committed primary-CV summary records:

outer_policy = cold_drug
cv_scope = frozen_outer_training_partition_only
outer_test_partition_used = false
label_column = interaction_kd_le_1000_nM
n_splits = 5
random_state = 20260830

Each drug must occur in exactly one validation fold. The program verifies zero
drug overlap between every fold's train and validation subsets. The 14 outer
holdout drugs and their outcome values are not selected for this analysis.

Fixed Candidate Models

Every fold uses the same four configurations already compared in the primary
inner-CV experiment:

dummy_prior — empirical class-prevalence baseline.

logistic_regression_balanced — standardized L2 logistic regression with
balanced class weights.

random_forest_balanced — 300 trees, depth 12, minimum leaf size 5,
square-root feature sampling, balanced class weights, and train-fold-only
zero-variance removal.

hist_gradient_boosting_balanced — regularized histogram gradient boosting
with 200 iterations, depth 3, leaf size 20, no internal early stopping, and
train-fold-only zero-variance removal.

The random seed for a fit is always 20260830 + fold_index. No configuration
is altered for the 100 nM label.

The previously selected primary model remains
random_forest_balanced, selected by unweighted mean inner-fold average
precision at 1,000 nM. This sensitivity stage records all fixed candidates for
context but does not re-rank or replace that decision.

Metrics and Reporting

Average precision (PR-AUC) is the principal imbalance-aware metric. ROC-AUC is
secondary. Accuracy, precision, recall, F1, and the threshold-0.5 confusion
matrix are supplementary operating-point summaries.

For each candidate and label variant, the report records:

class counts and prevalence;

five fixed-fold metric records;

unweighted fold mean, standard deviation, minimum, and maximum;

pooled OOF metrics, labelled descriptive only;

fixed model parameters and random-state rule.

No p-value, causal claim, biological-mechanism claim, or statement of model
superiority is supported by this analysis.

Outputs

Version-controlled summary:
reports/davis_threshold_sensitivity.json

Local ignored OOF predictions:
data/interim/davis_threshold_sensitivity_oof_predictions.csv

Reproduction Command

From project_02_drug_target:

& .\.venv\Scripts\python.exe -m src.models.threshold_sensitivity `
  --feature-table .\data\processed\davis_pair_features.csv `
  --inner-cv-summary .\reports\davis_inner_cold_drug_cv.json `
  --inner-oof-predictions .\data\interim\davis_inner_cold_drug_oof_predictions.csv `
  --n-splits 5 `
  --random-state 20260830 `
  --summary-output .\reports\davis_threshold_sensitivity.json `
  --oof-output .\data\interim\davis_threshold_sensitivity_oof_predictions.csv

Interpretation Guardrails

A more stringent 100 nM label is not a more correct label; it is a different,
deliberately stricter task.

A change in performance can reflect prevalence, label difficulty, or limited
sample size as well as model behaviour.

The results do not validate binding experimentally and cannot establish
biological causality or clinical utility.

No result from this post-selection sensitivity analysis may be used to tune
the candidate models or reopen the frozen outer holdout.