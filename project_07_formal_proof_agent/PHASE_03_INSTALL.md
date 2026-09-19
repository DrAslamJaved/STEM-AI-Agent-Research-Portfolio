# Install Phase 03 — Evaluation Harness

Extract this archive into the canonical repository root:

`G:\Research\STEM\Micro1_STEM_Agent_Portfolio`

The archive adds only Phase 03 paths under `project_07_formal_proof_agent`; it
does not contain a benchmark clone, Lean build cache, virtual environment, or
model outputs.

From PowerShell 7 in the canonical repository, run:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pytest -q `
  "$Project/tests/test_phase_01_contract.py" `
  "$Project/tests/test_phase_02_toolchain.py" `
  "$Project/tests/test_phase_03_evaluation_harness.py"

python "$Project/scripts/validate_phase_03_harness.py"
python "$Project/scripts/run_phase_03_synthetic_demo.py" --project-root $Project
Get-Content "$Project/results/phase_03_synthetic_demo.json"
```

Expected: all tests pass, the validator reports `PASS`, and the synthetic JSON
contains exactly one `compiled`, one `failed_compile`, and one
`rejected_static` result. It is harness evidence only—not a miniF2F score.

Stage the Phase 03 files and its compact synthetic evidence only after the
commands complete successfully. Do not add a local benchmark checkout, Lean
`_target` cache, virtual environment, or generated Python cache.
