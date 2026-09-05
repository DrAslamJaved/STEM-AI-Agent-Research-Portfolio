# Phase 07 — Final evaluation as a thin config-driven CLI dispatcher

## Decision

Implement the `evaluate --config configs/final.yaml` placeholder as a thin
CLI dispatcher (`cli.py`) that delegates everything -- config parsing, path
resolution, artifact-hash validation, trace freezing, policy comparison, and
paired-bootstrap confidence intervals -- to
`evidence_agent.evaluation.final` and `evidence_agent.evaluation.final_config`,
plus a new `evidence_agent.audit.bootstrap` module. `main()` is never called
recursively; `evaluate-citation-audit` and every other Phase 1-6 command
branch in `cli.py` is untouched.

## Why a separate module instead of reusing `evaluate-citation-audit`'s code

`evaluate-citation-audit` was explicitly off-limits for modification. Phase 07
needed the identical computation (load frozen artifacts, freeze a zero-
threshold trace, apply Phase 05 and the selected Phase 06 policy to that one
trace, load gold only after freezing) plus new behaviour (YAML config, hash
pinning, bootstrap CIs, a separate output set). Rather than importing private
helpers out of `cli.py`, Phase 07 reimplements this orchestration against the
same public library functions (`run_verification_agent`,
`apply_citation_audit_to_traces`, `evaluate_verification_traces`,
`load_gold_claim_annotations`, `load_calibration_report`,
`load_selected_policy`) that `evaluate-citation-audit` already uses. Both
commands therefore call the same underlying, unmodified Phase 05/06 code,
without either command depending on the other.

## Config safety

`configs/final.yaml` is loaded with `yaml.safe_load`, schema-validated against
`evidence_agent_final_evaluation_config_v1`, and every path is resolved
relative to `configs/`'s own directory -- not the process working directory.
Loading refuses a config whose any output path would resolve to
`results/citation_audit_dev.json` or
`results/citation_audit_cross_validation.json`. Five artifacts (corpus,
`claims_dev`, BM25 index, verifier model, calibration report) are required
and SHA-256-checked before any file is loaded; `train_claims` is accepted as
optional additional provenance and cross-checked against the verifier
bundle's own recorded training-claims hash when supplied.

## Paired-bootstrap confidence intervals

A new `evidence_agent.audit.bootstrap` module decomposes each policy's
audited, frozen decisions into per-claim primitives (`ClaimOutcome`), stripped
of claim id so that the aggregator can re-namespace every sentence/citation
key by its position in a bootstrap draw. This is required: a claim drawn
twice in one resample must contribute two distinct set entries, or its
duplicate would silently vanish under ordinary set deduplication. Every
metric -- including the two F1 scores and claim macro-F1 -- is recomputed
from each resample's *pooled* counts and labels, never by averaging per-claim
F1 values (which is a different, and wrong, quantity). Both policies are
resampled with the identical claim-id draw per replicate (paired design), so
any difference in a resampled metric comes from the policy, not from
inconsistent resampling.

## Verified against the already-committed Phase 06 result

Before writing any new document, the Phase 07 pipeline was run against the
exact same frozen artifacts Phase 06 used (`results/citation_audit_dev.json`'s
own `artifacts`/`model` block), and its
`comparison_to_phase_05_policy` block was checked byte-for-byte against
`results/citation_audit_dev.json`'s. All six deltas matched exactly,
confirming the reimplementation reproduces Phase 06's computation rather than
silently diverging from it.

## Outcome

Result JSON: `results/final_evaluation_dev.json`
Result JSON SHA-256: `b8f997142a49c3cf497ae48727f5378d91288c237562c5cce1a1b861060e03fd`
(this hash moves on every run because the JSON embeds wall-clock
`runtime_timing`; the raw trace hash below does not)
Raw trace SHA-256: `3a86c9fd96d99fe64c4c4e2b0fbdd5a2155ab02b8515cfd59d859e1cf821b5ae`

The bootstrap confirms only one clearly non-zero effect: the selected policy's
coverage is 0.373 lower than Phase 05's (95% CI [-0.430, -0.320], entirely
negative). Every intended quality improvement (unsupported-assertion rate,
faithfulness, citation-correctness F1, claim macro-F1) has a 95% CI that
contains zero at 300 development claims. See `reports/phase_07_final_evaluation.md`
for the full table and the trade-off framing.

## Evidence required before commit

1. Unit tests for config schema validation, config-relative path resolution,
   and refusal to target a Phase 06 result path (`tests/test_final_evaluation.py`).
2. Parametrized tests for a SHA-256 mismatch on each of the five required
   artifacts.
3. A structural test proving `load_gold_claim_annotations` is not called
   before the raw trace file exists on disk.
4. A test proving both policies are applied to the identical frozen claim
   order.
5. Deterministic paired-bootstrap tests (same seed -> same intervals; interval
   structure; rejection of mismatched claim populations).
6. An end-to-end CLI test running the full pipeline on a small fixture.
7. Passing tests (`76 passed`), `compileall`, and a fresh run against the real
   frozen SciFact artifacts, with its `comparison_to_phase_05_policy`
   cross-checked against the committed Phase 06 result.

## Reproducibility correction: project-relative path serialization

`results/final_evaluation_dev.json` originally recorded absolute,
machine-specific Windows paths (e.g. `G:\Research\...`) inside
`artifacts.*.path`, `trace_artifact.path`, `config_path`, and `output.*`,
because those fields were populated with `str(some_resolved_path)`. This broke
reproducibility of the committed JSON's *content* across machines even though
every hash and metric inside it was already correct and portable.

Fix, scoped to output serialization only (no change to metrics, policy
selection, bootstrap logic, artifact-hash validation, or command names):

- `FinalEvaluationConfig` gained a `project_root: Path` field
  (`final_config.py`), computed the same way the existing Phase-06-result
  guard already does (`config_dir.parent`, i.e. one level above `configs/`)
  -- derived from the config file's own location, never the process working
  directory.
- `final.py` gained `_relative_posix_path(path, project_root)`, which renders
  any path inside `project_root` as a `path.relative_to(project_root).as_posix()`
  string (forward slashes, no drive letter), falling back to an absolute
  POSIX string only for a path declared outside the project. Every path
  written into the report dict (`artifacts.*.path`, `trace_artifact.path`,
  `config_path`, `output.*`) now goes through this helper.
- `run_evaluate_command`'s returned `result_path`/`trace_path` (used by the
  CLI to print where the file actually is, and by callers to open it) are
  sourced from `config.output.*` directly rather than from the now-relative
  `report["output"]` block, so the command's actual file-location contract is
  unchanged -- only the JSON's own recorded provenance strings became
  project-relative.
- Added `test_run_evaluate_command_records_project_relative_posix_paths`,
  asserting every path field in the result JSON is relative, uses forward
  slashes, and matches no Windows drive-letter pattern, and that the frozen
  trace is still reachable by rejoining the recorded relative path onto the
  project root.
- `results/final_evaluation_dev.json` and `reports/phase_07_final_evaluation.md`
  were regenerated. The raw trace SHA-256 is unchanged
  (`3a86c9fd96d99fe64c4c4e2b0fbdd5a2155ab02b8515cfd59d859e1cf821b5ae`),
  confirming the evaluation itself did not change; only the result JSON's own
  hash moved (to `b8f997142a49c3cf497ae48727f5378d91288c237562c5cce1a1b861060e03fd`)
  because its path fields' text changed (and `runtime_timing` moves on every
  run regardless).
- Verified with the worktree-root virtual environment
  (`../.venv/Scripts/python.exe`): `pytest --basetemp artifacts/pytest_phase07_portability -q`
  (77 passed), `compileall src tests`, a fresh `evaluate --config
  configs/final.yaml` run, and `git diff --check`.
