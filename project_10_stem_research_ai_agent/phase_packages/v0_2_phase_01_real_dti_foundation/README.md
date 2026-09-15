# v0.2 Increment 1 — Real Davis DTI Foundation

This increment adds a non-networked Davis loader (including the canonical
binary-pickle `Y` matrix), source hash capture, pKd
derivation, deterministic long-form records, a formal data contract, and unit
tests. It intentionally does **not** download a dataset or train a model.
Those actions require an approved source copy and are the next increment.

## Apply on a dedicated branch

```powershell
$repo = "G:\Research\STEM\Micro1_STEM_Agent_Portfolio"
Set-Location $repo
git pull --ff-only origin main
git switch -c feature/project-10-v0-2-real-dti-foundation
```

Extract this package into `$repo`, then run:

```powershell
Set-Location "$repo\project_10_stem_research_ai_agent"
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q .\tests
```

After placing the approved Davis files in `data/raw/davis/`, copy the example
manifest to `config/v0_2_davis_manifest.json`, calculate and approve its hashes,
complete its values, and run:

```powershell
python .\experiments\run_v0_2_data_contract_demo.py `
  --data-dir .\data\raw\davis `
  --output .\results\v0_2_davis_data_report.json
```

Do not commit raw data. Commit the completed manifest and the generated report
only after independently reviewing the source and its licence/terms.
