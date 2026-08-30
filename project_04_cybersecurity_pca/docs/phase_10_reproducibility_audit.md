# Phase 10 Reproducibility Audit

## Status

Audit completed on 30 August 2026. Project closure evidence is ready for
pull-request review and merge. This audit does not alter the Phase 9
non-deployment recommendation.

## Scope

The audit covers the tracked Project 04 source code, tests, configuration,
results, and GitHub Actions workflow. It does not claim independent
reproduction of the official raw UNSW-NB15 files, which are intentionally
ignored and unavailable to the public CI runner.

## Audited revision

- branch: feature/project-04-phase-10-closure;
- validated commit: 099d4476dc266d11c2812106728223d614d99949;
- workflow: Project 04 CI;
- GitHub Actions run:
  https://github.com/DrAslamJaved/STEM-AI-Agent-Research-Portfolio/actions/runs/33305506914.

## Local Windows evidence

- Python: 3.12.8;
- full validation: 627 passed;
- combined branch coverage: 95.10%;
- required coverage threshold: 90%;
- dependency check: passed;
- package import: passed;
- Git diff whitespace check: passed.

## Clean GitHub Actions evidence

- runner: Ubuntu 24.04;
- Python: 3.12.14;
- package installation: passed;
- full validation: 626 passed, 1 skipped;
- combined branch coverage: 95.10%;
- required coverage threshold: 90%;
- workflow conclusion: success;
- uploaded artifact: project-04-validation-python-3.12;
- artifact ID: 9730326344;
- artifact SHA-256:
  078a0ed00d9109d6c9681d9c6af8d32d8bc4030359e35988e89ce584af6f346e.

## Expected data-dependent skip

tests/test_unsw_integration.py is skipped on the clean CI runner because the
official ignored UNSW-NB15 raw files are not committed to the repository.
This is expected and does not bypass the committed-data validation suite.

## Cross-platform repair

The first Linux run exposed a platform-dependent SHA-256 assertion over raw
floating-point reconstruction-error bytes. The test was revised to verify
the alignment contract directly: labels, predictions, scenarios, and
reconstruction errors must correspond exactly to the same flow IDs. This is
a stronger behavioural contract and avoids falsely requiring byte-identical
floating-point results across operating systems.

## Reproduction command

~~~
python -m pytest -q --cov=cyber_pca --cov-branch --cov-fail-under=90 --cov-report=term-missing
~~~

## Closure decision

The tracked Project 04 implementation reproduces successfully in both the
validated Windows environment and a clean Linux GitHub Actions environment.
The project is ready for final pull-request review. The PCA baseline remains
an untuned research baseline and is not recommended for operational
cybersecurity deployment.
