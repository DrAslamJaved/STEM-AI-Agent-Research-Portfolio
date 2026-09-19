# Install Phase 04 — LLM-Only Baseline

Extract this archive into the canonical repository root:

`G:\Research\STEM\Micro1_STEM_Agent_Portfolio`

It adds only Phase 04 paths beneath `project_07_formal_proof_agent`. It does
not contain a miniF2F clone, Lean build cache, virtual environment, API key, or
real model output.

From PowerShell 7 in the canonical repository, run:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pytest -q `
  "$Project/tests/test_phase_01_contract.py" `
  "$Project/tests/test_phase_02_toolchain.py" `
  "$Project/tests/test_phase_03_evaluation_harness.py" `
  "$Project/tests/test_phase_04_llm_only_baseline.py"

python "$Project/scripts/validate_phase_04_baseline.py"
python "$Project/scripts/run_phase_04_synthetic_baseline.py" --project-root $Project

Get-Content "$Project/results/phase_04_synthetic_baseline.json"
```

Expected: **25 tests pass**, the validator reports `PASS`, and the synthetic
evidence reports `compile_at_1: 0.5` with observed `pass@2: 1.0`. These are
synthetic control values, not miniF2F or LLM-performance results.

Stage only the Phase 04 source, policy, prompt, synthetic task fixture, tests,
and compact generated evidence. Do not add benchmark clones, Lean `_target`
caches, virtual environments, API credentials, or Python cache directories.
