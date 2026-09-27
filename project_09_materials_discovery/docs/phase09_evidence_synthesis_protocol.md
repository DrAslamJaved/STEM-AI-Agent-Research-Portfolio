# Phase 09: Evidence synthesis and release protocol

## Purpose

Phase 09 produces a claim-limited, reproducible synthesis of the completed
Project 09 evidence. It does not train a new model, select labels, tune a
threshold, or alter any earlier result. The synthesis hashes every supplied
artifact and revalidates the machine-readable evidence before rendering a
portfolio-ready report.

## Required evidence

* Phase 03 official checkpoint JSON: the predeclared active-learning comparison.
* Phase 08 deterministic official JSON: the structure descriptor-versus-graph
  comparison.

Optional paired inputs are accepted only together:

* Phase 05 checkpoint and prediction-record JSON files for exploratory
  disagreement/error and conditional-coverage diagnostics.
* Phase 06 checkpoint and prediction-record JSON files for normalized-conformal
  diagnostics.

The CLI rejects an incomplete optional pair. Existing Phase 05 and Phase 06
validators check that prediction-level records reproduce their saved checkpoint
metrics before Phase 09 summarizes them.

## Claim policy

The synthesis distinguishes three statements:

1. The Phase 03 ensemble threshold result is predeclared. Missing crossings
   remain missing, and a non-positive paired difference is not label saving.
2. Phase 05 and Phase 06 subgroup and calibration comparisons are exploratory;
   they cannot revise the Phase 03 acquisition conclusion.
3. Phase 08 is a fixed CPU model comparison. A graph model that does not beat
   its descriptor comparator does not justify a later graph-UQ or
   graph-acquisition claim.

All analyses are retrospective benchmarks with hidden pool labels during
selection. They are not prospective laboratory discovery.

## Reproduction command

Run from `project_09_materials_discovery` after placing the reviewed JSON files
under the ignored `results/` directory:

```powershell
.\.venv\Scripts\python.exe -m p09.synthesis `
  --phase03 results\phase03_official.json `
  --phase08 results\phase08_official_deterministic.json `
  --phase05-checkpoints results\phase05_official.json `
  --phase05-predictions results\phase05_predictions.json `
  --phase06-checkpoints results\phase06_official.json `
  --phase06-predictions results\phase06_predictions.json `
  --output results\phase09_evidence_synthesis.md `
  --manifest results\phase09_evidence_manifest.json
```

The Markdown report and manifest are review artifacts and remain ignored by
Git. The manifest records filenames, byte counts, SHA-256 values, validated
summary metrics, and explicit claim boundaries.
