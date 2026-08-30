# Phase 7 Prompt: UNSW-NB15 Acquisition and Leakage-Safe Preprocessing

## Objective

Acquire the official curated UNSW-NB15 training, testing, and feature-description
files; validate their provenance, integrity, schema, and encoding; then construct
deterministic leakage-safe model matrices for the later real-data experiment.

## Authoritative source

Use only the official UNSW Research dataset page:

https://research.unsw.edu.au/projects/unsw-nb15-dataset

The official files are manually downloaded and stored under `data/raw`. Raw
files are immutable, ignored by Git, and must never be redistributed through
the repository.

## Required raw files

- `UNSW_NB15_training-set.csv`;
- `UNSW_NB15_testing-set.csv`;
- `NUSW-NB15_features.csv`.

Preserve the official descriptor filename exactly. The curated files use UTF-8.
The descriptor uses Windows-1252 (`cp1252`).

## Acquisition and schema requirements

1. Calculate SHA-256 and byte size for every raw file.
2. Validate 175,341 training observations and 82,332 testing observations.
3. Require the identical verified 45-column curated schema.
4. Validate 49 feature-description rows and the four descriptor columns.
5. Reject duplicate rows, duplicate partition-local IDs, missing values,
   nonnumeric numeric fields, nonfinite values, invalid binary labels, and
   attack-category/label inconsistencies.
6. Treat `(source_partition, id)` as the record key because `id` overlaps
   between the two official partitions.
7. Write deterministic tracked provenance under `reports/validation`.

## Leakage-safe preprocessing requirements

1. Use only normal records from the official training file for development.
2. Split those normal records deterministically into 75% fitting and 25%
   calibration partitions with random seed 42.
3. Exclude every training attack record.
4. Exclude `id`, `attack_cat`, and `label` from model inputs.
5. Retain 39 numeric inputs.
6. Fit `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` only on
   normal fitting traffic for `proto`, `service`, and `state`.
7. Combine numeric inputs with the learned one-hot inputs.
8. Reject normal-fit zero-variance features.
9. Fit `StandardScaler` only on the combined normal fitting matrix.
10. Transform normal calibration and official test features only after the
    encoder and scaler are frozen.
11. Preserve globally unique `flow_id` values derived from partition and ID.
12. Do not use official test labels or attack categories for fitting.

## Verification requirements

Use red-green TDD for public interfaces, loading, validation, manifest writing,
splitting, encoding, standardization, evidence writing, package exports, and
documentation. Include boundary tests and an optional integration test that
skips when ignored raw files are unavailable.

Run the complete regression suite with line and branch coverage, compile all
source and test files, verify dependencies, validate JSON and XML evidence,
and run `git diff --check`.

## Phase boundary

Phase 7 must not fit PCA, select components, calculate reconstruction errors,
calibrate an anomaly threshold, produce anomaly predictions, or report
UNSW-NB15 performance. Those operations belong to Phase 8.
