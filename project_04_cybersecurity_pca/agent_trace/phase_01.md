# Phase 01 Agent Trace

## Date

2026-08-27

## Scope

Establish the Project 4 repository foundation, isolated Python environment,
package configuration, research protocol, and mathematical contract.

No PCA algorithm, cybersecurity dataset, or numerical experiment was
implemented during these initial steps.

## Git starting evidence

```text
Branch: feature/project-04-cyber-pca
Local-only commits: 0
Remote-only commits: 0
Starting commit: 6a8441d
Starting commit message: Add continuous integration workflow
```

The untracked sibling directory `project_02_drug_target` is outside the
Project 4 scope and was not modified.

## Environment evidence

```text
Operating system: Windows
Python: 3.12.8
Project environment: .venv
Package: cyber-pca 0.1.0
```

Installed dependencies:

```text
numpy: 2.5.2
pandas: 2.3.3
scikit-learn: 1.9.0
matplotlib: 3.11.1
seaborn: 0.13.2
PyYAML: 6.0.3
joblib: 1.5.3
pytest: 9.1.1
pytest-cov: 7.1.0
```

Dependency validation:

```text
python -m pip check
No broken requirements found.
```

Package-import validation:

```text
Package imported successfully
Version: 0.1.0
Path: src/cyber_pca/__init__.py
```

## Completed work

- created the Project 4 feature branch;
- created an isolated Python 3.12 environment;
- created the initial repository architecture;
- added raw and derived data exclusion rules;
- configured the Python package;
- installed runtime and development dependencies;
- verified the package import;
- documented the research protocol;
- documented the mathematical contract;
- documented the critical PCA reasoning challenge.

## Work not yet executed

The following remain `TO BE EXECUTED/VERIFIED`:

- manual PCA implementation;
- mathematical unit tests;
- coverage measurement;
- synthetic cybersecurity data;
- reconstruction-error anomaly detector;
- threshold calibration;
- classification metrics;
- UNSW-NB15 acquisition;
- real-data evaluation;
- continuous-integration execution.

No test result, coverage percentage, anomaly metric, or dataset result has been
claimed.

## Manual PCA Stage 1: Input and covariance

### Red test

Executed:

```text
python -m pytest tests\test_pca_manual.py -q --tb=short
```

Actual result before implementation:

```text
ModuleNotFoundError: No module named 'cyber_pca.pca_manual'
1 collection error
Exit code: 2
```

This was the expected test-driven-development failure because the PCA module
did not exist.

### Implementation

Created:

```text
src/cyber_pca/pca_manual.py
```

Implemented:

- two-dimensional numeric input validation;
- minimum observation validation;
- finite-value validation;
- component-count validation;
- conversion to float64;
- feature centering;
- sample covariance using \(X_c^TX_c/(n-1)\).

### Green test

Actual result:

```text
16 passed in 1.10s
Exit code: 0
```

Compilation result:

```text
python -m compileall src\cyber_pca tests\test_pca_manual.py
Exit code: 0
```

Independent covariance demonstration:

```text
Means: [3.         4.66666667]
Covariance:
[[4.         6.        ]
 [6.         9.33333333]]
Symmetric: True
```

Manual PCA remains incomplete. Eigendecomposition, transformation,
reconstruction, and anomaly detection are still `TO BE EXECUTED/VERIFIED`.

## Manual PCA Stage 2: Eigendecomposition

### Red test

The eigendecomposition contract was defined before implementation.

Actual result:

```text
7 failed, 16 passed in 1.51s
Exit code: 1
```

All seven failures were expected missing-attribute errors for:

- `explained_variance_`;
- `components_`;
- `explained_variance_ratio_`;
- `cumulative_explained_variance_`.

### Implementation

Implemented:

- symmetric eigendecomposition using `numpy.linalg.eigh`;
- descending eigenvalue ordering;
- corresponding eigenvector ordering;
- deterministic eigenvector sign orientation;
- scale-aware numerical tolerance;
- harmless negative-eigenvalue clipping;
- explained-variance ratios;
- cumulative explained variance;
- retained-component selection.

### Green test

```text
23 passed in 0.89s
Exit code: 0
```

Compilation:

```text
Exit code: 0
```

Independent numerical evidence:

```text
Largest eigenpair residual: 4.440892098500626e-16
Orthonormality error: 6.661338147750939e-16
Final cumulative explained variance: 1.0
```

Transformation and reconstruction remain `TO BE EXECUTED/VERIFIED`.

## Manual PCA Stage 3: Transformation and reconstruction

### Red test

```text
9 failed, 23 passed in 1.43s
Exit code: 1
```

The failures were expected because `transform`, `inverse_transform`,
`reconstruct`, `reconstruction_error`, and `fit_transform` were not yet
implemented.

### Implementation

Implemented:

- principal-component transformation;
- inverse transformation;
- reconstruction;
- observation-level mean squared reconstruction error;
- combined fit and transform;
- fitted-model validation;
- feature-count validation;
- component-score-count validation.

### Green test

```text
32 passed in 0.34s
Exit code: 0
```

Compilation:

```text
Exit code: 0
```

Reconstruction demonstration:

```text
One-component mean error: 0.024415062755840305
Two-component mean error: 0.008488767606951553
Full-component maximum error: 5.577493118945435e-31
```

The aggregate reconstruction error decreased as components were added, and
full-component reconstruction was numerically exact.

## Manual PCA Stage 4: Numerical and reference validation

Added validation for:

- principal-subspace agreement with scikit-learn;
- explained-variance agreement with scikit-learn;
- near-collinear numerical stability;
- deterministic repeated fitting;
- invalid eigenvalue tolerances;
- zero-variance input;
- single-observation transformation.

### Command-entry correction

The following command was entered incorrectly:

```text
cpython -m pytest tests\test_pca_manual.py -q --tb=short
```

PowerShell reported:

```text
The term 'cpython' is not recognized
```

No test was executed by that command. The displayed `$LASTEXITCODE` was stale
and was not treated as test evidence.

The command was corrected to:

```text
python -m pytest tests\test_pca_manual.py -q --tb=short
```

### Focused test result

```text
41 passed in 21.72s
PowerShell command succeeded: True
Exit code: 0
```

### Coverage result

```text
41 passed in 4.28s
Total coverage: 96.95%
Required coverage: 90%
Coverage gate: passed
Exit code: 0
```

Generated evidence:

```text
reports/validation/phase_01_pytest.xml
reports/validation/phase_01_coverage.xml
```

Manual PCA is now validated for covariance, eigendecomposition,
transformation, reconstruction, numerical stability, and reference-subspace
agreement.

## Manual PCA Stage 5: Independent mathematical validation

### Red test

```text
ModuleNotFoundError: No module named 'cyber_pca.validation'
1 error during collection
Exit code: 2
```

### Implementation

Created:

```text
src/cyber_pca/validation.py
tests/test_math_validation.py
```

Implemented independent checks for:

- covariance symmetry;
- descending eigenvalues;
- eigenpair consistency;
- eigenvector orthonormality;
- nonnegative eigenvalues;
- explained-variance consistency;
- full-component reconstruction.

### Full-suite result

```text
48 passed in 14.71s
Exit code: 0
```

Compilation:

```text
python -m compileall src tests
Exit code: 0
```

Independent report:

```text
Status: passed
Maximum absolute reconstruction error: 1.7763568394002505e-15
Explained-variance ratio sum: 1.0
```

All seven independent mathematical checks passed.

## Phase 1 CLI: Mathematical validation command

### Red test

Before implementation:

```text
ModuleNotFoundError: No module named 'cyber_pca.cli'
1 error during collection
Exit code: 2
```

The module command also correctly failed before `__main__.py` existed:

```text
No module named cyber_pca.__main__
```

### Implementation

Created:

```text
src/cyber_pca/cli.py
src/cyber_pca/__main__.py
tests/test_cli.py
```

Added the console entry point:

```text
cyber-pca = cyber_pca.cli:main
```

Implemented:

- command help;
- deterministic validation matrix;
- `validate-math`;
- `--dry-run`;
- configurable JSON output;
- passing/failing exit codes;
- module execution through `python -m cyber_pca`.

### Green tests

```text
CLI tests: 7 passed in 1.45s
Complete suite: 55 passed in 18.88s
Exit code: 0
```

Compilation:

```text
python -m compileall src tests
Exit code: 0
```

## Phase 1 CLI execution evidence

Module help:

```text
python -m cyber_pca --help
Exit code: 0
```

Installed console help:

```text
cyber-pca --help
Exit code: 0
```

Dry-run evidence:

```text
Dry-run exit code: 0
Dry-run file exists: False
```

Real validation:

```text
Mathematical validation: PASSED
Checks passed: 7 / 7
Validation exit code: 0
JSON parsing: successful
Explained-variance ratio sum: 1.0
```

Deterministic report evidence:

```text
First hash:  DD4A361841051A42FA77BE6062DE0FB5C3BF1D7B935C4FD796755A1ABF3A012C
Second hash: DD4A361841051A42FA77BE6062DE0FB5C3BF1D7B935C4FD796755A1ABF3A012C
Hashes match: True
Second-run exit code: 0
```

Final coverage execution:

```text
55 passed in 23.28s
Total coverage: 93.21%
Required coverage: 90%
Coverage gate: passed
Exit code: 0
```

Generated evidence:

```text
reports/validation/math_validation.json
reports/validation/phase_01_final_pytest.xml
reports/validation/phase_01_final_coverage.xml
```

## Final Phase 1 verification

Final test command:

```text
python -m pytest
  --cov=cyber_pca
  --cov-report=term-missing
  --cov-report=xml:reports/validation/phase_01_final_coverage.xml
  --cov-fail-under=90
  --junitxml=reports/validation/phase_01_final_pytest.xml
  -W error::DeprecationWarning
```

Actual result:

```text
58 tests passed in 30.46s
Total coverage: 93.26%
Required coverage: 90%
Coverage gate: passed
Exit code: 0
```

Compilation:

```text
python -m compileall src tests
Exit code: 0
```

Final mathematical validation:

```text
Status: passed
Checks passed: 7 / 7
Maximum reconstruction error: 8.881784197001252e-16
Explained-variance ratio sum: 1.0
```

Configuration:

```text
Random seed: 42
Explained-variance target: 0.95
Calibration quantile: 0.99
Final dataset: UNSW-NB15
```

## Phase 1 completion status

Completed:

- transparent covariance-based PCA;
- symmetric eigenvalue decomposition;
- explained-variance calculations;
- principal-component transformation;
- inverse transformation and reconstruction;
- observation-level reconstruction error;
- input and numerical-stability validation;
- scikit-learn principal-subspace comparison;
- independent mathematical validation;
- dry-run and executable CLI;
- deterministic JSON validation evidence;
- automated tests and coverage evidence.

Remaining for later phases:

- deterministic synthetic cybersecurity observations;
- interpretable synthetic attack families;
- leakage-safe preprocessing;
- normal-only threshold calibration;
- anomaly classification and evaluation metrics;
- UNSW-NB15 acquisition and validation;
- real-data experiment;
- agent reasoning rubric;
- continuous-integration execution for Project 4.

Phase 1 contains no claimed cybersecurity detection performance.

## Git hygiene correction

The first staged whitespace validation reported:

```text
tests/test_pca_manual.py:474: trailing whitespace
Staged diff-check exit code: 2
```

The trailing spaces were removed without changing program behaviour. The
corrected file was restaged.

Final result:

```text
git diff --cached --check
Exit code: 0
```

No Project 2 file, virtual environment, cache, compiled Python file, or package
build artifact was intentionally staged.