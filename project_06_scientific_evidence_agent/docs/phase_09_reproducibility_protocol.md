# Phase 09 — Reproducibility handoff protocol

## Purpose

Phase 09 proves that the Phase 06-08 evidence already committed to this
repository is reproducible from a clean environment, with a single command.
It does **not** retrain the verifier, rebuild retrieval, recalibrate the
citation-audit policy, re-run the controlled experiment, or download the raw
SciFact release. Those operations belong to earlier phases and remain
untouched. Phase 09 only *verifies*.

## What the one-command gate verifies

`scripts/run_phase09_reproducibility.py` is the single entry point:

```powershell
& .\.venv\Scripts\python.exe .\scripts\run_phase09_reproducibility.py --clean
```

It derives the project root from its own file location (never the caller's
current working directory), then:

1. optionally removes -- and only ever removes -- the one directory it
   manages, `artifacts/phase09_reproducibility/`, when `--clean` is passed;
2. creates a brand-new, isolated virtual environment under that same managed
   directory;
3. installs the exact locked third-party dependencies from
   `requirements-dev.lock`, then installs this project editable with
   `pip install --no-deps -e .` so dependency resolution never re-runs;
4. runs `pip check`;
5. runs the full pytest suite with a JUnit XML report and branch coverage in
   XML, JSON, and terminal form;
6. runs `python -m compileall` over `src` and `tests`;
7. runs the Phase 09 evidence-integrity gate:
   `evidence_agent reproducibility --config configs/reproducibility.yaml
   --output-dir artifacts/phase09_reproducibility/manifest`.

The gate itself (`evidence_agent reproducibility`) is a separate, lighter
check with its own CLI command. It reads only committed, Git-tracked files --
the three frozen Phase 06-08 result JSON files, their Markdown reports, and
their agent traces -- and confirms:

- each frozen result's SHA-256 matches the digest declared in
  `configs/reproducibility.yaml`;
- Phase 07 (`final_evaluation_dev.json`) and Phase 08
  (`controlled_experiments_dev.json`) both declare
  `evaluation_label: held_out_development_evaluation` and
  `is_independent_test: false`;
- Phase 08's adversarial evaluator-regression suite
  (`adversarial_evaluator_suite.all_passed`) is `true`;
- each frozen result's own SHA-256 is echoed inside both its committed
  Markdown report and its committed agent trace, so the narrative and the
  machine record cannot silently drift apart;
- every `path` / `*_path` field recorded inside those frozen results is a
  portable, project-relative path -- never an absolute path or a Windows
  drive-letter path that would leak a contributor's local machine layout.

Every path in `configs/reproducibility.yaml` resolves relative to the config
file itself, never the shell's current working directory, so the gate behaves
identically regardless of where it is invoked from.

## What the gate intentionally does not do

- It does not download the raw SciFact release, or use anything under
  `data/raw/`.
- It does not train, retrain, or recalibrate any model or policy.
- It does not re-run the Phase 07 or Phase 08 experiments; it only checks the
  already-frozen results committed from those runs.
- It does not overwrite any Phase 06-08 result, report, config, or agent
  trace. It writes only inside the directory named by `--output-dir` (the
  clean-environment runner always points this beneath
  `artifacts/phase09_reproducibility/`, which is Git-ignored).
- It does not require the ignored raw dataset, a trained model artifact, or a
  full runtime trace -- everything it reads is already committed.

## Frozen Phase 06-08 evidence checked by the gate

| Result | Path | SHA-256 |
| --- | --- | --- |
| Phase 06 cross-validated policy selection | `results/citation_audit_cross_validation.json` | `27447a85cff918b5f1b322cf0598983c435143caec7a56484049a89e163f12b3` |
| Phase 07 held-out development evaluation | `results/final_evaluation_dev.json` | `b8f997142a49c3cf497ae48727f5378d91288c237562c5cce1a1b861060e03fd` |
| Phase 08 controlled experiment | `results/controlled_experiments_dev.json` | `32fe3b1d7c6886db580b08ad9ffb694f85961bd7b19cafb980e1507a2c081b5d` |

These digests are declared once in `configs/reproducibility.yaml` and must
never be edited to match a changed file; a changed file means the frozen
evidence was altered, which Phase 09 must fail loudly.

## Held-out development, not an independent test

Phase 07 and Phase 08 are development evaluations against SciFact's public
development split. Both explicitly label themselves
`held_out_development_evaluation` with `is_independent_test: false`, and
`reports/figures/phase_09_evaluation_tradeoff.svg` repeats that label on the
figure itself. Any future independent test would require a held-out split
that was never touched during retrieval tuning, verifier training, or policy
calibration -- SciFact's public development split does not qualify, since
Phase 06's calibration folds were drawn from the same claim population.

## CI evidence

`.github/workflows/project06-reproducibility.yml` runs the same
`scripts/run_phase09_reproducibility.py --clean` command on Python 3.12 for
every pull request and push touching `project_06_scientific_evidence_agent/**`
or the workflow file itself. It uploads the JUnit XML, the coverage XML and
JSON reports, and the Phase 09 reproducibility manifest as workflow
artifacts. CI never downloads SciFact, trains a model, or writes to a
committed `results/`, `reports/`, or `agent_trace/` path -- all of its output
lives under the same `artifacts/phase09_reproducibility/` tree the local
runner uses, which is discarded with the runner rather than committed.

## Final handoff materials

- `results/phase_09_reproducibility.json` -- the machine-readable record of
  the last full clean-environment run: Python version, test count, measured
  coverage, and the gate manifest.
- `reports/phase_09_reproducibility.md` -- the narrative report, including
  the result JSON's own SHA-256 (computed after the file was written, never
  self-hashed inside it).
- `agent_trace/phase_09_reproducibility.md` -- the execution trace for this
  phase's own work.
- `reports/figures/phase_09_evaluation_tradeoff.svg` -- a deterministic,
  dependency-free SVG rendering of the Phase 08 audited-agent-minus-direct-RAG
  trade-off, generated from the frozen Phase 08 result by
  `scripts/generate_phase09_figure.py`.
