# Davis Model-Input Protocol

## Purpose

This stage prepares leakage-audited feature matrices for model fitting.
It does not create a new split, fit a transformer, resample classes, train a
classifier, tune hyperparameters, or evaluate predictions.

## Input Artifacts

The model-input loader uses only locally generated and Git-ignored artifacts:

- `data/processed/davis_pair_features.csv`
- `data/interim/davis_split_assignments.csv`

The split-assignment table was generated previously with seed `20260830`.
It is treated as frozen during model development.

## Join Key

`observed_pair_index` is the sole join key between the feature table and split
assignments.

The loader requires:

- one unique observed-pair index in the feature table;
- one assignment per observed-pair index for the selected policy;
- no missing or extra assignment indices;
- exactly two partitions named `train` and `test`.

## Feature Matrix

The model matrix `X` contains exactly `FEATURE_COLUMNS` imported from
`src.features.representations`.

It contains 36 transparent drug and target descriptors.

The following fields are excluded from `X`:

- `drug_id`;
- `target_id`;
- `observed_pair_index`;
- matrix indices;
- affinity values;
- pKd;
- binary labels;
- split-policy and partition labels.

Identifiers are retained only in separate metadata tables for audits and later
error analysis.

## Labels

The initial binary outcome is:

`interaction_kd_le_1000_nM`

This corresponds to the pre-specified operational threshold:

`Kd <= 1,000 nM`

The stricter `Kd <= 100 nM` label remains a sensitivity analysis and must use
the same feature and split logic. It must not be selected because it produces a
more favorable model score.

## Leakage Safeguards

The loader never calls a split function. It uses only the stored assignments.

It rejects:

- missing, duplicate, or extra pair assignments;
- invalid partitions;
- missing or non-finite feature values;
- non-binary labels;
- partitions without both classes;
- overlapping drugs in `cold_drug`;
- overlapping targets in `cold_target`.

For `random_pair`, drug and target overlap is expected and must be reported as
an interpolation benchmark rather than evidence of unseen-entity
generalization.

## Primary Holdout

The headline holdout is the fixed `cold_drug` partition:

- training pairs: 23,868;
- test pairs: 6,188;
- training drugs: 54;
- test drugs: 14;
- drug overlap: 0;
- target overlap: 442.

The selected outer fold is fold 4 of the pre-specified five-fold
`StratifiedGroupKFold` design.

## Audit Output

`reports/davis_model_input_audit.json` records the selected policy, label,
feature columns, class counts, partition sizes, unique entities, and entity
overlap counts.

## Interpretation Boundary

A valid model-input audit establishes only that the benchmark data were
prepared consistently with the stated split policy. It does not demonstrate
predictive utility, statistical significance, biological mechanism, clinical
utility, or causal effects.