# Project 07 — Phase 02 Package

This package adds the reproducibility contract for the frozen miniF2F benchmark and its Lean 3 toolchain.

## Install

Extract this archive at the root of the active Project 07 worktree:

```text
G:\Research\STEM\Micro1_STEM_Agent_Portfolio_project07
```

It adds files beneath `project_07_formal_proof_agent/` only. It does not contain the benchmark checkout, generated logs, or the generated validation JSON; retain the locally generated files already present in the worktree.

Then run, from PowerShell 7:

```powershell
Set-Location 'G:\Research\STEM\Micro1_STEM_Agent_Portfolio_project07'
& .\project_07_formal_proof_agent\scripts\apply_phase_02_ignores.ps1
python -m pytest -q project_07_formal_proof_agent\tests\test_phase_01_contract.py project_07_formal_proof_agent\tests\test_phase_02_toolchain.py
python project_07_formal_proof_agent\scripts\validate_phase_02_toolchain.py --project-root project_07_formal_proof_agent
```

Only after all checks pass should the Phase 02 files be staged. The local upstream clone and `.log` files must remain untracked.
