# Phase 09 reproducibility execution trace

1. Wrote `src/evidence_agent/reproducibility.py`, a dedicated module (not
   `cli.py`) implementing the Phase 9 evidence-integrity gate: SHA-256
   validation of three frozen Phase 06-08 results, held-out-development label
   checks, the Phase 08 adversarial-suite guard, report/agent-trace hash
   consistency, and rejection of absolute or drive-letter provenance paths.
2. Wrote `configs/reproducibility.yaml`, whose paths resolve relative to the
   config file itself, declaring the three frozen result hashes given for
   this phase and matched against the real committed files before anything
   else was written.
3. Added the thin `evidence_agent reproducibility --config --output-dir` CLI
   dispatcher entry in `cli.py`; all logic stayed in `reproducibility.py`.
4. Discovered that the real, frozen `results/citation_audit_cross_validation.json`
   already contains relative, backslash-separated Windows-style paths (written
   by an earlier Phase 06 run on Windows). Per the immutability constraint on
   Phase 6-8 artifacts, the provenance check was scoped to reject only
   POSIX-absolute, drive-letter, and UNC paths -- not relative backslash
   paths -- so the gate does not fail against evidence it must not alter.
5. Added `tests/test_reproducibility.py` (config-relative resolution, each
   frozen-hash mismatch, held-out-development and adversarial-suite guards,
   report/agent-trace consistency, absolute/drive-letter path rejection,
   output-directory safety, and CLI end-to-end execution on a temporary
   fixture) and confirmed the real, committed `configs/reproducibility.yaml`
   also passes against the actual repository evidence.
6. Added `scripts/run_phase09_reproducibility.py`, a stdlib-only, one-command
   clean-environment runner that derives its project root from its own file
   location, manages only `artifacts/phase09_reproducibility/`, and refuses
   to remove any other path (`_safe_clean`). Added
   `tests/test_phase09_scripts.py` covering that guard, path derivation, and
   the requirements-dev.lock format.
7. Generated `requirements-dev.lock` from the actual working Phase 9 virtual
   environment (`pip freeze`, minus the local editable project line) and
   proved a brand-new isolated environment installs it cleanly.
8. Added `scripts/generate_phase09_figure.py` and ran it against the frozen,
   hash-verified `results/controlled_experiments_dev.json` to produce
   `reports/figures/phase_09_evaluation_tradeoff.svg` -- a deterministic,
   dependency-free SVG showing coverage and official abstract/sentence F1 for
   both arms, their deltas and bootstrap confidence intervals, and the
   disclaimer "Held-out development evaluation -- not an independent test."
9. Added `.github/workflows/project06-reproducibility.yml`, triggered on
   pushes and pull requests touching this project or the workflow file,
   running the same one-command runner on Python 3.12 and uploading the
   JUnit XML, coverage XML/JSON, and reproducibility manifest as workflow
   artifacts.
10. Ran the full validation sequence: `pip install -e ".[dev]"`, the full
    pytest suite, `compileall`, the reproducibility CLI check against the
    real repository, and finally
    `scripts/run_phase09_reproducibility.py --clean` end to end from a
    completely fresh virtual environment. All steps passed: 149 tests passed
    (0 failed/errored/skipped) in 34.15 s under Python 3.12.8, with 84.97%
    combined statement+branch coverage.
11. Wrote `results/phase_09_reproducibility.json` from those measured
    numbers, computed its SHA-256, and recorded it in this trace and in
    `reports/phase_09_reproducibility.md`.
12. **Correction:** a follow-up review of `_is_forbidden_committed_path` found
    that `PureWindowsPath(value).is_absolute()` returns `False` for a
    Windows-rooted path with no drive letter (e.g. `\rooted\file.json`),
    since `is_absolute()` requires both a drive and a root -- so that one
    dangerous shape was silently accepted. Replaced the check with explicit,
    unambiguous rules: reject an empty path, a path starting with `/`
    (POSIX-absolute), a path starting with `\` (covers both a Windows-rooted
    path and a UNC path, since a UNC path is just a rooted path with two
    leading backslashes), or a drive-letter path (`_DRIVE_LETTER_PATTERN`);
    accept everything else, including a relative backslash path. Removed the
    now-unused `_UNC_PATTERN` constant and `PureWindowsPath` import, and
    tightened the module docstring's provenance bullet to name all four
    rejected shapes. No SHA-256, label, adversarial-suite, or report/trace
    consistency check was touched.
13. Added direct unit tests of `_is_forbidden_committed_path` in
    `tests/test_reproducibility.py` for every accepted shape (relative POSIX,
    relative backslash, and a dedicated regression test for a relative
    backslash path) and every rejected shape (POSIX-absolute x2,
    drive-letter x2, drive-letter-without-root, Windows-rooted, UNC, and
    empty), and extended the existing end-to-end gate test to also cover the
    two newly-caught shapes through the full config/JSON pipeline. This
    raised the suite from 149 to 164 tests.
14. Re-ran `scripts/run_phase09_reproducibility.py --clean` end to end from a
    completely fresh virtual environment after the correction. All steps
    passed: 164 tests passed (0 failed/errored/skipped) in 32.13 s under
    Python 3.12.8, with 85.10% combined statement+branch coverage.
15. Refreshed `results/phase_09_reproducibility.json` with those corrected
    measured numbers (test_summary and coverage_summary only; every other
    field, including the frozen-evidence hashes and gate manifest, was
    already correct and unchanged), computed its new SHA-256, and updated
    that value here and in `reports/phase_09_reproducibility.md`. No file
    hashes itself.
16. Verified no Phase 06-08 result, report, config, or agent trace was
    modified, and that no source code, CI workflow, dependency lock, figure,
    or test file changed beyond the path-policy fix and its own regression
    tests: `git status --short` shows only the Phase 9 files and the
    original `README.md` / `src/evidence_agent/cli.py` edits.
17. Fixed a cross-platform checksum defect: the three frozen Phase 06-08
    result JSON digests declared in `configs/reproducibility.yaml` were
    recorded from a Windows CRLF checkout, so a Linux CI checkout of the same
    committed content (LF line endings) failed the raw-byte SHA-256
    comparison even though nothing about the evidence had changed. Added
    `sha256_frozen_result_json` (and its `_canonicalize_line_endings` helper)
    to `src/evidence_agent/reproducibility.py`: it normalizes CRLF and lone-CR
    line endings to LF, re-expands to a canonical CRLF byte representation,
    then hashes that -- giving the same digest on Windows and Linux. The
    generic `sha256_file` helper other phases use, the three declared
    Phase 06-08 SHA-256 values, and every Phase 06-08 result/report/config/
    agent-trace file were left untouched; only `_validate_frozen_result_hashes`
    was repointed at the new helper. Added two regression tests -- LF/CRLF
    equivalence, and that a real content change still changes the digest --
    plus confirmed the existing end-to-end test against the real committed
    `configs/reproducibility.yaml` still passes. Documented the fix in
    `docs/phase_09_reproducibility_protocol.md`.
18. Re-ran `scripts/run_phase09_reproducibility.py --clean` end to end from a
    completely fresh virtual environment after the checksum fix. All steps
    passed: 166 tests passed (0 failed/errored/skipped) in 27.03 s under
    Python 3.12.8, with 85.12% combined statement+branch coverage -- the two
    new cross-platform line-ending checksum regression tests raised the suite
    from 164 to 166.
19. Refreshed `results/phase_09_reproducibility.json` with those measured
    numbers (test_summary and coverage_summary, plus a note that the suite
    now includes the cross-platform line-ending checksum regression tests);
    every other field, including the three frozen-evidence hashes and the
    gate manifest, was already correct and unchanged. Computed its new
    SHA-256 and updated that value here and in
    `reports/phase_09_reproducibility.md`. No file hashes itself.

Result JSON SHA-256: `311bc37dcecbfd97625023f74ef6c4aee5ceadc416e62aa230d5981e8aa055b1`
