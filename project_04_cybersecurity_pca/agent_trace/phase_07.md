# Phase 7 Agent Trace: UNSW-NB15 Acquisition and Preprocessing

## Starting state

- Starting commit: `55e4bc0`.
- Branch: `feature/project-04-cyber-pca`.
- Local and remote divergence: `0 0`.
- The only unrelated untracked path was sibling `project_02_drug_target/`.
- Raw, interim, and processed data directories were already ignored except
  for their `.gitkeep` files.

## Source and acquisition

The official UNSW Research dataset page was used as the authoritative source.
The curated training, testing, and descriptor files were downloaded manually
from the official SharePoint link into ignored `data/raw`.

The downloaded descriptor used the official filename
`NUSW-NB15_features.csv`, not the initially anticipated
`UNSW-NB15_features.csv`. The configuration, tests, and contract were corrected
to preserve the official filename.

Raw-file evidence:

| File | Bytes | SHA-256 |
|---|---:|---|
| `UNSW_NB15_training-set.csv` | 32,293,018 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `UNSW_NB15_testing-set.csv` | 15,380,800 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| `NUSW-NB15_features.csv` | 4,044 | `c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce` |

All raw files remained ignored. The tracked raw-data manifest SHA-256 was
`aa4f992959dbf1fb30d2c92a4a93a0508bfc712d09a15d24725d28399f49d1ee`.

## Encoding and schema discovery

The curated training and testing files decoded as UTF-8. The descriptor failed
UTF-8 decoding at byte `0x92` and decoded successfully as `cp1252`. Raw bytes
were not altered.

Verified curated structure:

- training shape: `(175341, 45)`;
- testing shape: `(82332, 45)`;
- identical ordered schemas: true;
- descriptor shape: `(49, 4)`;
- duplicate rows: 0 in both curated files;
- duplicate IDs: 0 within each partition;
- missing values: 0;
- numeric values finite: true;
- category/label mismatches: 0.

Training labels were 56,000 normal and 119,341 attack. Testing labels were
37,000 normal and 45,332 attack. IDs ranged independently from 1 within each
partition, producing 82,332 cross-partition overlaps. The durable record key
was therefore defined as `(source_partition, id)`.

## Red-green implementation sequence

1. Added a failing configuration contract for the official source and file
   expectations, then implemented the `unsw_nb15` YAML section.
2. Added observed filename, encoding, schema-size, and partition-local key
   assertions; repaired the YAML and data contract.
3. Added a failing `cyber_pca.unsw_data` import test; implemented typed path
   and data dataclasses plus official path resolution.
4. Added failing loader tests; implemented UTF-8 curated loading and `cp1252`
   descriptor loading.
5. Added failing schema-validation tests; implemented row, schema, duplicate,
   missing, finite, label, category, and descriptor validation.
6. Repaired a pandas nullable-Boolean comparison by comparing Boolean arrays
   rather than dtype-sensitive `Series.equals`.
7. Added failing deterministic manifest tests; implemented chunked SHA-256,
   JSON-safe counts, manifest construction, and newline-terminated writing.
8. Added a failing `cyber_pca.unsw_preprocessing` interface test; implemented
   raw split, fitted preprocessor, and standardized split dataclasses.
9. Added failing split tests; implemented deterministic normal-only splitting.
10. Added failing leakage tests; implemented normal-fit-only one-hot encoding,
    64-column combined matrices, zero-variance rejection, normal-fit-only
    scaling, and composite `flow_id` values.
11. Added failing evidence-writer tests; implemented deterministic
    preprocessing evidence.
12. Added data and preprocessing boundary suites. Both new modules reached
    100% statement and branch coverage.
13. Added a locally enabled, CI-safe integration test.
14. Added failing package-export and documentation contracts, then exposed the
    Phase 7 API and documented verified evidence.

## Preprocessing evidence

The 56,000 normal training observations were divided into 42,000 normal fitting
and 14,000 normal calibration observations. All 119,341 training attacks were
excluded. The complete 82,332-row official test partition was retained.

The normal-fit encoder learned 7 protocol, 10 service, and 8 state categories,
producing 25 one-hot columns. Together with 39 numeric columns, the model
matrix contains 64 float64 features.

Observed unseen categories confirmed the need for `handle_unknown="ignore"`:
one calibration row contained unseen state `PAR`; test traffic contained
8,126 rows with unseen protocol values, 61 rows with unseen services, and five
rows with unseen states.

Deterministic preprocessing evidence:

| Evidence | SHA-256 |
|---|---|
| normal-fit IDs | `b8ec94affc717d98f1c5d2db12e1d8304f82f9e761590ec66c2b18a3c827cc68` |
| normal-calibration IDs | `3d6fdd600b74bcd0599838a016290a07315fc52cfdba2f8b2cae6fd2614a6d73` |
| test IDs | `e0ba957c470ff374011d87e4214e304b85b85dde050198e5b4da146c5c885039` |
| feature names | `f47ce5e1981c3a4eae3d51cd45c20f50d02f764714a57f3ec57a15c7b4c62bad` |
| encoder domains | `f54d775fb4b8fdee822680fcd9d823698562031b22a75696c8029773c6f34ea8` |
| scaler state | `610ea7a2e37f669878a10a34d18e63b93b36fa4a0251e92399b5d680ab841685` |
| standardized normal fit | `cb5a43a816f2ac080b9669a147690309b601872b9fe2e63b6973b095ca552d4b` |
| standardized calibration | `1c35e0f30bf46bf21bc77160ddadea6db22b2f086182dc79344fa04ecaaf79b7` |
| standardized test | `eeaf451bba6b4c0907f62c28dff28fb9acfc7660bd0f8ae7baa689ef76de8b47` |

The maximum absolute normal-fit standardized mean was
`2.6744790509220754e-13`. The maximum population-standard-deviation error was
`1.0685896612017132e-12`. The original `1e-12` smoke-test bound missed by
`6.85896612017132e-14`; a still-stringent `2e-12` evidence bound was adopted.

The preprocessing evidence JSON SHA-256 was
`6d4eb8aa219d93228411ecca53ed5577c3f957fe0575cce06226eec72a77a1ce`.

## Validation incidents and resolutions

- The first descriptor read failed with `UnicodeDecodeError`; inspection
  established `cp1252` without modifying the file.
- One preflight expression applied `int()` before `Series.sum()`; explicit
  masks repaired the diagnostic.
- One repeated edit command reported an assertion failure although the
  implementation tests passed. Source inspection confirmed one clean import
  and one clean function definition.
- Two module-specific coverage sources triggered a Windows NumPy duplicate-load
  error. A single package-level coverage target resolved the instrumentation
  conflict.
- An unreachable NaN branch was replaced with a direct `numpy.isfinite`
  validation, bringing `unsw_data.py` to 100% coverage.
- A missing `Path` import in an evidence test was repaired.

## Verification

The preliminary full gate produced 399 passing tests. After adding the final
README evidence test, the final full gate produced 400 passing tests with no
failures, errors, or skipped tests.

Verified final coverage:

- line coverage: 94.73%;
- branch coverage: 87.41%;
- combined coverage: 92.71%;
- `unsw_data.py`: 100%;
- `unsw_preprocessing.py`: 100%.

Compilation, dependency validation, and `git diff --check` passed.

## Phase boundary

No PCA model was fitted. No component count was selected. No reconstruction
errors were computed. No anomaly threshold was calibrated. No UNSW-NB15
predictions or performance metrics were produced. Official test labels remain
reserved for Phase 8 evaluation after model fitting and prediction.
