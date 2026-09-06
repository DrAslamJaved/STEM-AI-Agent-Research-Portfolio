# Phase 09 — Reproducibility handoff

**This phase verifies committed Phase 06-08 evidence; it does not retrain,
recalibrate, or re-run any earlier experiment.**

## What the one-command gate verifies

```powershell
& .\.venv\Scripts\python.exe .\scripts\run_phase09_reproducibility.py --clean
```

From a project root derived from the script's own file location, this single
command creates an isolated virtual environment under
`artifacts/phase09_reproducibility/`, installs the exact locked dependencies
from `requirements-dev.lock`, installs this project editable with
`pip install --no-deps -e .`, runs `pip check`, the full pytest suite with
JUnit XML and branch coverage, `python -m compileall`, and finally the
Phase 09 evidence-integrity gate
(`evidence_agent reproducibility --config configs/reproducibility.yaml`).
That gate confirms, by reading only committed `results/`, `reports/`, and
`agent_trace/` files:

- the SHA-256 of each of the three frozen Phase 06-08 results;
- that Phase 07 and Phase 08 both declare
  `evaluation_label: held_out_development_evaluation` and
  `is_independent_test: false`;
- that Phase 08's adversarial evaluator-regression suite passed
  (`adversarial_evaluator_suite.all_passed == true`);
- that each frozen result's own SHA-256 is echoed inside both its committed
  report and its committed agent trace;
- that every provenance path recorded inside those frozen results is
  project-relative, never absolute or Windows drive-lettered.

## What it intentionally does not do

- No raw-data download: it never touches `data/raw/` or the SciFact release.
- No retraining, recalibration, or re-execution of the Phase 07 or Phase 08
  experiments -- it checks their already-frozen, committed outputs only.
- No overwriting of any prior result, report, config, or agent trace; the
  gate writes only inside the directory named by `--output-dir`.

## Measured results of the last full clean-environment run

These numbers reflect the cross-platform checksum fix in
`sha256_frozen_result_json` (`src/evidence_agent/reproducibility.py`), which
verifies each frozen Phase 06-08 result JSON's SHA-256 from a line-ending-
normalized canonical byte representation instead of raw bytes, so the gate
passes identically on a Windows CRLF checkout and a Linux LF checkout. The
suite now includes two dedicated regression tests for this fix --
`test_sha256_frozen_result_json_is_identical_for_lf_and_crlf_checkouts` and
`test_sha256_frozen_result_json_still_detects_real_content_changes` -- which
raised the suite from 164 to 166 tests. The generic `sha256_file` helper used
by every other phase, and all three frozen Phase 06-08 declared SHA-256
values below, are unchanged.

| Metric | Value |
| --- | --- |
| Python version | 3.12.8 |
| Isolated environment | fresh venv under `artifacts/phase09_reproducibility/venv`, built only from `requirements-dev.lock` + editable install |
| `pip check` | No broken requirements found |
| Tests collected | 166 |
| Tests passed / failed / errored / skipped | 166 / 0 / 0 / 0 |
| Test duration | 27.03 s |
| Coverage (statements + branches) | 85.12% (2,741 / 3,067 statements plus partial branch credit) |
| Coverage (statements only) | 89.37% |
| Coverage (branches only) | 69.47% |

Result JSON SHA-256: `311bc37dcecbfd97625023f74ef6c4aee5ceadc416e62aa230d5981e8aa055b1`

(This hash was computed after `results/phase_09_reproducibility.json` was
written; the file does not hash itself.)

## Frozen Phase 06-08 evidence checked

| Result | Path | SHA-256 |
| --- | --- | --- |
| Phase 06 cross-validated policy selection | `results/citation_audit_cross_validation.json` | `27447a85cff918b5f1b322cf0598983c435143caec7a56484049a89e163f12b3` |
| Phase 07 held-out development evaluation | `results/final_evaluation_dev.json` | `b8f997142a49c3cf497ae48727f5378d91288c237562c5cce1a1b861060e03fd` |
| Phase 08 controlled experiment | `results/controlled_experiments_dev.json` | `32fe3b1d7c6886db580b08ad9ffb694f85961bd7b19cafb980e1507a2c081b5d` |

## CI evidence

`.github/workflows/project06-reproducibility.yml` runs the identical
`scripts/run_phase09_reproducibility.py --clean` command on Python 3.12 for
every pull request and push touching `project_06_scientific_evidence_agent/**`
or the workflow file. It uploads the JUnit XML, coverage XML, coverage JSON,
and the Phase 09 reproducibility manifest as workflow artifacts:

- `artifacts/phase09_reproducibility/pytest_junit.xml`
- `artifacts/phase09_reproducibility/coverage.xml`
- `artifacts/phase09_reproducibility/coverage.json`
- `artifacts/phase09_reproducibility/manifest/reproducibility_manifest.json`

CI never downloads SciFact, trains a model, or writes to a committed
`results/`, `reports/`, or `agent_trace/` path -- every workflow output lives
under the same ignored `artifacts/phase09_reproducibility/` tree the local
runner uses.

## Held-out development, not an independent test

Phase 07 and Phase 08 remain held-out *development* evaluations, not
independent tests: `reports/figures/phase_09_evaluation_tradeoff.svg` states
this on the figure itself, and the Phase 09 gate fails if either result's
`evaluation_label` or `is_independent_test` field is ever edited to claim
otherwise.
