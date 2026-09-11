# STEM Research AI Agent — MVP

This project is a researcher-controlled MVP for reproducible STEM research
workflows. Its demonstration domain combines drug--target interaction (DTI)
prediction with fuzzy-similarity analysis.

## Delivered workflow

1. Research contract and researcher inputs.
2. DTI and fuzzy-data validation with provenance controls.
3. Reproducible DTI baselines, evaluation, and fuzzy similarity measures.
4. Claim-level evidence ledger and source-verification statuses.
5. Controlled manuscript-section drafting constrained by approved evidence.
6. Draft quality and originality-risk audits.
7. Append-only human review, revision, approval, and locking controls.
8. Release evaluation combining evidence, audit, review, and reproducibility gates.

## Run the MVP checks

```powershell
$env:PYTHONPATH = "$PWD\\src"
python -m pytest -q .\\tests
python -m compileall -q .\\src
python .\\experiments\\run_phase_08_release_demo.py
```

## Responsible-use boundary

The agent assists a researcher; it does not autonomously author, verify, or
submit scientific work. Evidence verification, citation checking, originality
assessment, interpretation, and publication decisions remain subject to human
review.
