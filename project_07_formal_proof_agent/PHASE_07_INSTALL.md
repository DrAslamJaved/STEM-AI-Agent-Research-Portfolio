# Install Phase 07 — Fuzzy Similarity Formalization (R3)

Extract this archive into the canonical repository root:

`G:\Research\STEM\Micro1_STEM_Agent_Portfolio`

This corrected archive supersedes the initial and R2 Phase 07 packages after
pinned Lean 3 / Windows diagnostics. Extract with `-Force` to replace only the
Phase 07 overlay files beneath `project_07_formal_proof_agent`. It does not
contain a miniF2F clone, Lean build cache, virtual environment, API key,
compiler evidence, or model output.

From PowerShell 7 in the canonical repository, run:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pytest -q `
  "$Project/tests/test_phase_01_contract.py" `
  "$Project/tests/test_phase_02_toolchain.py" `
  "$Project/tests/test_phase_03_evaluation_harness.py" `
  "$Project/tests/test_phase_04_llm_only_baseline.py" `
  "$Project/tests/test_phase_05_sympy_counterexamples.py" `
  "$Project/tests/test_phase_06_lean_repair.py" `
  "$Project/tests/test_phase_07_fuzzy_formalization.py"

python "$Project/scripts/validate_phase_07_fuzzy_formalization.py"
python "$Project/scripts/run_phase_07_formalization_manifest.py" --project-root $Project
```

Expected: **66 tests pass**, the validator reports `PASS`, and the manifest is
written.  The manifest is source provenance only, not a proof result.

Compile the unchanged Lean source through the Phase 02 pinned checkout:

```powershell
$Elan = Join-Path $env:USERPROFILE '.elan\bin\elan.exe'
$LeanCwdCandidates = @(
  (Join-Path $Project 'benchmarks\minif2f\upstream'),
  'G:\Research\STEM\Micro1_STEM_Agent_Portfolio_project07\project_07_formal_proof_agent\benchmarks\minif2f\upstream'
)
$LeanCwd = $LeanCwdCandidates |
  Where-Object { Test-Path (Join-Path $_ 'leanpkg.toml') } |
  Select-Object -First 1

if ($null -eq $LeanCwd) {
  throw 'Pinned miniF2F v1 checkout is absent. Restore the Phase 02 checkout; do not substitute another revision.'
}

python "$Project/scripts/compile_phase_07_lean.py" `
  --project-root $Project `
  --elan $Elan `
  --lean-cwd $LeanCwd

Get-Content "$Project/results/phase_07_lean_compile.json"
```

Expected: `status` is `compiled`, `compilation_passed` is `true`, and the
compiler exit code is `0`.  If the compile command fails, do not stage Phase 07
yet; paste its complete output so the source can be corrected while preserving
the pinned versions.

After a successful compile, stage only the listed Phase 07 source, policy,
prompt, scripts, tests, manifest, and compiler evidence.  Do not add benchmark
clones, Lean `_target` caches, virtual environments, credentials, Python cache
directories, or this ZIP file.
