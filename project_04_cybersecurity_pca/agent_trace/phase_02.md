# Phase 02 Agent Trace

## Date

2026-08-27

## Starting state

Branch: `feature/project-04-cyber-pca`

Starting commit: `c8c60c7`

Starting commit title: `Validate Phase 1 Markdown integrity`

Project 2 remained untracked and outside the Phase 2 scope.

## Scope

Implement and validate a deterministic synthetic cybersecurity network-flow
dataset containing normal traffic and four attack scenarios.

No preprocessing, PCA anomaly fitting, threshold calibration, or UNSW-NB15
experiment was implemented in this phase.

## Dataset-contract correction

The first contract-file attempt was made from the repository root rather than
the Project 4 directory.

Observed outcome:

- the file was created under the repository-root `docs` directory;
- pytest could not find `tests/test_package.py`;
- no tests ran;
- exit code: 4.

The displayed fence count of 76 was caused by PowerShell interpreting backticks
inside a double-quoted Python command. Inspection showed that the contract
content itself was correct.

The file was moved to
`project_04_cybersecurity_pca/docs/synthetic_data_contract.md`.

Corrected focused result:

- tests passed: 5;
- failures: 0;
- exit code: 0.

## Configuration red-green cycle

Red result:

- failing test: `test_baseline_configuration_contract`;
- cause: missing `synthetic_data` configuration;
- failure: `KeyError`;
- tests passed: 4;
- tests failed: 1;
- exit code: 1.

The synthetic dataset defaults, attack types, feature columns, and metadata
columns were then added to `configs/baseline.yaml`.

Green result:

- tests passed: 5;
- failures: 0;
- exit code: 0.

## Generator import red test

The first generator test failed during collection because
`cyber_pca.synthetic_data` did not exist.

Observed result:

- collection errors: 1;
- exception: `ModuleNotFoundError`;
- exit code: 2.

## Generator skeleton

A module containing constants and an explicit unimplemented generator was
created.

Observed intermediate result:

- tests passed: 1;
- tests failed: 2;
- exception: `NotImplementedError`;
- exit code: 1.

## First generator implementation

The deterministic generator was implemented using
`numpy.random.default_rng`.

Observed result:

- focused tests passed: 3;
- failures: 0;
- exit code: 0;
- generated shape: 5,000 rows and 13 columns;
- duplicate flow identifiers: 0;
- missing values: 0;
- model-feature dtype: float64.

Default scenario counts were:

- normal: 4,000;
- port scan: 250;
- denial of service: 250;
- brute force: 250;
- exfiltration: 250.

## Observed attack medians

Selected median values from the default seed-42 dataset were:

| Scenario | Connection rate | Unique destination ports | Failed logins | Outbound bytes |
|---|---:|---:|---:|---:|
| Normal | 1.44 | 3.00 | 0.0 | 8,051.55 |
| Port scan | 9.30 | 58.63 | 0.0 | 1,785.56 |
| Denial of service | 21.52 | 3.00 | 0.0 | 49,290.04 |
| Brute force | 6.42 | 3.00 | 17.0 | 7,739.83 |
| Exfiltration | 1.45 | 3.00 | 0.0 | 139,753.85 |

These observed values were used to design conservative median-based signature
tests.

## Expanded generator validation

The expanded synthetic-data suite tested schema, data quality, deterministic
execution, attack signatures, shuffling, duplicates, and invalid arguments.

Observed result:

- tests passed: 24;
- failures: 0;
- exit code: 0.

## Package-export red-green cycle

Red result:

- collection errors: 1;
- cause: synthetic-data symbols were not exported by `cyber_pca`;
- exception: `ImportError`;
- exit code: 2.

The public package interface was updated.

Green result:

- package tests passed: 5;
- failures: 0;
- exit code: 0.

## Complete regression evidence

Observed complete-suite result:

- tests collected: 84;
- tests passed: 84;
- failures: 0;
- exit code: 0;
- compilation exit code: 0.

The generated Phase 2 XML artifacts reported:

- tests: 84;
- failures: 0;
- errors: 0;
- total coverage: 94.82%.

The final coverage-command exit code will be recorded after the final evidence
rerun.

## Determinism evidence

Seed-42 SHA-256:

`35005389b137bd472e44b44c987597b1b7e13b8fa88a4c099c110c50986e1561`

Repeated seed-42 SHA-256:

`35005389b137bd472e44b44c987597b1b7e13b8fa88a4c099c110c50986e1561`

Seed-43 SHA-256:

`259439881e9176e08d735911ef5f8d7b753340b20fc5f658776c69720e6bc76e`

Observed conclusions:

- same-seed hashes matched;
- different-seed hashes differed.

## Pending completion checks

The following remain pending before the Phase 2 commit:

- README status update;
- final coverage rerun with captured exit code;
- final documentation regression check;
- staged scope verification;
- commit and push.

## Final Phase 2 validation

The README was updated to mark deterministic synthetic cybersecurity data
as completed and leakage-safe preprocessing as the next phase.

Final verified results:

- tests collected: 84;
- tests passed: 84;
- failures: 0;
- test duration: 12.52 seconds;
- total coverage: 94.82%;
- required coverage: 90%;
- coverage gate: passed;
- coverage exit code: 0;
- compilation exit code: 0;
- diff check: clean.

Staging, commit, push, and remote verification remain pending.
