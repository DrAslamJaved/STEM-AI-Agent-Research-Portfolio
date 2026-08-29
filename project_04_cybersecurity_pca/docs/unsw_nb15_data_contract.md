# UNSW-NB15 Data Acquisition and Validation Contract

## 1. Purpose

Phase 7 introduces the first external cybersecurity dataset into Project 4.
Its purpose is to acquire, identify, validate, and prepare the official
UNSW-NB15 curated partitions without fitting PCA, calibrating a threshold, or
reporting real-data anomaly-detection performance.

## 2. Authoritative source

Dataset name: UNSW-NB15.

Publisher: UNSW Canberra at the Australian Defence Force Academy.

Official project page:

https://research.unsw.edu.au/projects/unsw-nb15-dataset

The official page describes network traffic containing normal activity and
nine attack categories. It reports 175,341 records in the curated training
partition and 82,332 records in the curated testing partition.

## 3. Academic-use condition

UNSW grants free use for academic research and requires citation of the
dataset publications identified on the official project page. The project
will preserve the official source URL and citation requirement in its
provenance evidence.

This repository does not redistribute the raw dataset.

## 4. Required source files

Phase 7 expects these files in `data/raw`:

- `UNSW_NB15_training-set.csv`;
- `UNSW_NB15_testing-set.csv`;
- `NUSW-NB15_features.csv`.

The two curated partitions are used instead of the four larger source CSV
files because they provide a documented and reproducible train-test boundary.

## 5. Repository boundary

All files under `data/raw` are ignored by Git except `.gitkeep`.

Raw CSV files must remain immutable after acquisition. Cleaning, transformation,
encoding, scaling, partitioning, and feature selection must not overwrite the
source files.

The tracked provenance manifest is written to:

`reports/validation/phase_07_unsw_nb15_manifest.json`

## 6. Provenance manifest

The manifest must record, for every acquired file:

- filename;
- official source page;
- acquisition method;
- acquisition timestamp in UTC;
- byte size;
- SHA-256 digest;
- row count when applicable;
- column count when applicable;
- ordered column names when applicable.

The manifest must also record the Python, pandas, NumPy, and scikit-learn
versions used during validation.

## 7. Schema validation

The curated training and testing CSV files must be validated independently
before they are combined with any workflow.

Validation must examine:

- exact row counts;
- column counts;
- ordered column names;
- data types;
- duplicate rows;
- duplicate identifiers;
- missing values;
- positive and negative infinity;
- label values;
- attack-category values;
- agreement between labels and attack categories;
- equality of training and testing schemas.

The observed schema will be recorded from the files themselves rather than
silently assumed.

## 8. Label and leakage boundary

The expected binary label column is `label`, where 0 represents normal traffic
and 1 represents attack traffic.

The expected attack-category column is `attack_cat`.

The expected record identifier is `id`.

The following columns must never be model features:

- `id`;
- `label`;
- `attack_cat`.

Official test labels and attack categories remain hidden until Phase 8
prediction is complete.

## 9. Categorical features

The expected categorical traffic descriptors are:

- `proto`;
- `service`;
- `state`.

Their observed categories must be recorded separately for the training and
testing partitions. Phase 7 does not yet fit an encoder.

Any Phase 8 encoder must be fitted using normal fitting traffic only and must
handle previously unseen test categories without refitting.

## 10. Normal-only development boundary

Only records with training label 0 are eligible for PCA fitting or threshold
calibration.

The normal records in the official training partition will later be split
deterministically into:

- 75% normal fitting traffic;
- 25% normal calibration traffic.

The official testing partition remains intact as the hidden-label evaluation
partition.

## 11. Validation failure policy

Phase 7 must stop with a clear error if:

- a required file is absent;
- a raw file is empty;
- a digest cannot be calculated;
- an expected row count differs;
- train and test columns differ;
- binary labels contain values other than 0 and 1;
- identifiers are missing or duplicated;
- required columns are absent;
- numeric columns contain infinity;
- labels conflict with attack categories.

Observed missing values or duplicate rows must be reported explicitly and
handled only through a documented, tested policy.

## 12. Phase boundary

Phase 7 may acquire data, validate provenance, inspect the schema, and create
reproducible preparation contracts.

Phase 7 must not:

- fit a scaler;
- fit PCA;
- select principal components;
- calculate a real-data threshold;
- generate real-data predictions;
- inspect real-data performance metrics.

Those operations belong to Phase 8.

## Observed official-file representation

The two curated observation files are UTF-8 CSV files. The official
`NUSW-NB15_features.csv` descriptor uses Windows-1252 (`cp1252`) encoding.
The raw bytes must remain unchanged; encoding is specified only when the
files are read.

The curated training and testing files each contain 45 columns. The feature
descriptor contains 49 rows and four descriptive columns. The descriptor
documents the wider source feature set, while the curated train and test
partitions expose the identical verified 45-column schema used by this
project.

The `id` field is unique within each curated partition but is not globally
unique across the training and testing files. Identifiers are therefore
partition-local. Any persisted or combined record must use the composite key
`(source_partition, id)` so that training and testing observations cannot be
mistaken for the same flow.

## Leakage-safe feature preparation

The curated model input contains 39 numeric columns and three categorical
columns: `proto`, `service`, and `state`. The identifier, attack category,
and binary label are excluded from model features.

Only normal observations from the official training partition may determine
the normal fitting and normal calibration membership. With seed 42 and a
75:25 split, the verified partitions contain 42,000 normal fitting records
and 14,000 normal calibration records. All 119,341 attack records in the
official training file are excluded from model development.

`OneHotEncoder` is fitted using normal fitting traffic only. Its verified
normal-fit vocabulary produces 25 encoded categorical columns. Unknown
categories are ignored during transformation instead of being learned from
calibration or testing traffic. The observed calibration partition contains
one row with an unseen `state`. The official test partition contains 8,126
rows with an unseen `proto`, 61 rows with an unseen `service`, and five rows
with an unseen `state`.

The 39 numeric columns and 25 encoded categorical columns form a 64-column
model matrix. `StandardScaler` is fitted to this combined matrix using normal
fitting traffic only. All verified normal-fit numeric and encoded columns
have positive variance, so the zero-variance policy remains rejection rather
than automatic feature deletion.

The official testing partition remains intact. Its features may be
transformed after the encoder and scaler are frozen, but its `label` and
`attack_cat` values are reserved for evaluation and must not influence
splitting, encoding, scaling, PCA fitting, component selection, or threshold
calibration.

## Phase 7 evidence artifacts

The tracked provenance manifest is
`reports/validation/phase_07_unsw_nb15_manifest.json`. The deterministic
preprocessing report is
`reports/validation/phase_07_unsw_nb15_preprocessing.json`. Pytest and coverage
evidence are stored in `reports/validation/phase_07_pytest.xml` and
`reports/validation/phase_07_coverage.xml`.

The raw CSV files remain ignored and are not repository artifacts.
