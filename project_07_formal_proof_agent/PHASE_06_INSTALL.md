# Install Phase 06 — Lean Generate–Verify–Repair

Extract this archive into the canonical repository root:

`G:\\Research\\STEM\\Micro1_STEM_Agent_Portfolio`

It adds only Phase 06 paths beneath `project_07_formal_proof_agent`. It does
not contain a miniF2F clone, Lean build cache, virtual environment, API key, or
real model output.

From PowerShell 7 in the canonical repository, run:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pytest -q `
  "$Project/tests/test_phase_01_contract.py" `
  "$Project/tests/test_phase_02_toolchain.py" `
  "$Project/tests/test_phase_03_evaluation_harness.py" `
  "$Project/tests/test_phase_04_llm_only_baseline.py" `
  "$Project/tests/test_phase_05_sympy_counterexamples.py" `
  "$Project/tests/test_phase_06_lean_repair.py"

python "$Project/scripts/validate_phase_06_lean_repair.py"
python "$Project/scripts/run_phase_06_synthetic_lean_repair.py" --project-root $Project

Get-Content "$Project/results/phase_06_synthetic_lean_repair.json"
```

Expected: **48 tests pass**, the validator reports `PASS`, and the synthetic
evidence reports `compile_at_1: 0.5` and
`compile_after_lean_repair: 1.0`. These are workflow-control values, not
miniF2F, Lean benchmark, or model-performance results.

The real compiler factory uses the Phase 02 pin: miniF2F `v1` at
`f0dcc8b59e630fba00ba9569ca6714700e0a8801`, Lean 3.42.1, and the recorded
mathlib revision. It requires the already configured pinned miniF2F working
directory; do not substitute newer Lean, mathlib, or benchmark revisions.

Stage only the Phase 06 source, policy, prompt, synthetic fixture, tests, and
compact generated evidence. Do not add benchmark clones, Lean `_target` caches,
virtual environments, API credentials, Python cache directories, or the ZIP.
