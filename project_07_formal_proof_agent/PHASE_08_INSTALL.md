# Install Phase 08

Use only the Project 07 Phase 08 ZIP on branch
`feature/project-07-phase-08-evaluation-release`. Do not extract Project 10
Phase 08 materials into this repository.

From the canonical repository root:

```powershell
$Repo = 'G:\Research\STEM\Micro1_STEM_Agent_Portfolio'
$Zip = Join-Path $env:USERPROFILE 'Downloads\Project_07_Phase_08_Evaluation_Release.zip'

Set-Location $Repo
if ((git branch --show-current).Trim() -ne 'feature/project-07-phase-08-evaluation-release') {
    throw 'Stop: wrong branch.'
}
if (-not (Test-Path -LiteralPath $Zip)) {
    throw "Stop: Phase 08 ZIP is missing: $Zip"
}

Expand-Archive -LiteralPath $Zip -DestinationPath $Repo -Force
```

Run the focused verification and deterministic smoke fixture:

```powershell
$Project = 'project_07_formal_proof_agent'

python -m pytest -q "$Project/tests/test_phase_08_evaluation_release.py"
python "$Project/scripts/run_phase_08_synthetic_release.py" --project-root $Project
python "$Project/scripts/validate_phase_08_evaluation_release.py"
python -m compileall -q "$Project/src" "$Project/scripts"
```

The generated `phase_08_synthetic_release.json` must report
`release_ready: false`; that is the expected result because the fixture is
synthetic and has no human approval. It is not a miniF2F score.

Before committing, stage the Project 07 directory plus the dedicated GitHub
Actions workflow only. Do not stage ZIP files, benchmark clones, Lean caches,
virtual environments, raw candidate logs, or material from another project.
