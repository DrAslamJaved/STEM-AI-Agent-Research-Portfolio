# Install Phase 05 — SymPy Diagnostics and Exact Counterexamples

Extract this archive into the canonical repository root:

`G:\\Research\\STEM\\Micro1_STEM_Agent_Portfolio`

It adds only Phase 05 paths beneath `project_07_formal_proof_agent`. It does
not contain a miniF2F checkout, Lean build cache, virtual environment, API key,
or real model output.

From PowerShell 7 in the canonical repository, run:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pip install -r "$Project/requirements-phase05.txt"

python -m pytest -q `
  "$Project/tests/test_phase_01_contract.py" `
  "$Project/tests/test_phase_02_toolchain.py" `
  "$Project/tests/test_phase_03_evaluation_harness.py" `
  "$Project/tests/test_phase_04_llm_only_baseline.py" `
  "$Project/tests/test_phase_05_sympy_counterexamples.py"

python "$Project/scripts/validate_phase_05_sympy_workflow.py"
python "$Project/scripts/run_phase_05_synthetic_sympy_workflow.py" --project-root $Project

Get-Content "$Project/results/phase_05_synthetic_sympy_workflow.json"
```

Expected: **36 tests pass**, the validator reports `PASS`, and the synthetic
evidence reports `compile_at_1: 0.5` and
`compile_after_sympy_repair: 1.0`. Those values demonstrate workflow ordering
only; they are not miniF2F, Lean, or LLM-performance results.

Stage only the Phase 05 source, policy, prompt, fixture, tests, dependency
declaration, and compact generated evidence. Do not add benchmark clones, Lean
`_target` caches, virtual environments, API credentials, Python cache
directories, or the ZIP archive itself.
