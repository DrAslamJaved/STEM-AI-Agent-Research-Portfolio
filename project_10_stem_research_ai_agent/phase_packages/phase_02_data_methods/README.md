# Phase 02 — Data and Methodology Validation

## Goal
Validate supplied DTI and fuzzy-similarity data before analysis.

## Deliverables
- Immutable raw-data provenance manifest with SHA-256 hashes.
- DTI schema checks, missingness, duplicates, labels, and leakage-risk report.
- Fuzzy membership-vector validation: finite universe, values in [0,1], non-empty objects.
- `src/stem_research_agent/data_validation.py` and tests.

## Acceptance criteria
1. Invalid membership values and malformed DTI records are rejected.
2. Validation report records row counts, missingness, duplicates, and leakage warnings.
3. Raw data are not modified by the pipeline.

## Git checkpoint
`feat(project07): add data validation and provenance controls`
