# Davis Inner Cold-Drug Cross-Validation Protocol

## Purpose

This stage assesses the stability of fixed DTI model candidates across unseen
drugs while preserving the final cold-drug holdout partition.

It is an inner validation study, not a replacement for the fixed outer
14-drug test set.

## Nested Evaluation Structure

```text
All Davis pairs
├── Frozen outer cold-drug training partition: 54 drugs
│   └── Inner 5-fold StratifiedGroupKFold by drug_id
│       ├── Dummy prior baseline
│       ├── Weighted logistic regression
│       ├── Random Forest
│       └── Histogram gradient boosting
└── Frozen outer cold-drug test partition: 14 drugs
    └── Not accessed by inner CV

## Model-Selection Rule

The candidate with the highest unweighted mean inner-fold average precision is
selected for the primary post-selection analysis.

Pooled out-of-fold metrics are descriptive only. They do not determine model
selection because each fold uses an independently fitted model, whose
probabilities may not be calibrated on the same numerical scale.

With five grouped folds, fold-to-fold variation is descriptive evidence of
stability; it is not an independent-replicates hypothesis test and does not
establish statistical superiority between candidates.