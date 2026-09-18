# Phase 10 Amendment 01 — Source-location Binding

## Status

**APPROVED 2026-09-17.** This amendment changes only the permitted location
from which the already frozen Project 02 Davis artefacts may be read. It does
not change the data source, label, feature representation, compared methods,
split policy, tuning rule, outcomes, inference, or decision rule in the Phase
10 protocol.

## Reason for the amendment

The canonical portfolio's `project_02_drug_target/data` directory did not
contain the required data artefacts. A read-only search located the retained
Project 02 validation material in the validated Project 10 v0.2 worktree:

`G:\\Research\\STEM\\Project10_v02_clean`

The worktree retains Project 02 reports and validation records, while the
corresponding raw Davis source files occur under its Project 10 directory. This
amendment permits the runner to inspect that worktree as a source location;
it does not permit substitution of Project 10 v0.2's analysis regime.

## Frozen compatibility requirements

The amended source must provide all of the following before model execution:

1. Project 02 evidence that identifies the binary label
   `interaction_kd_le_1000_nM`.
2. A Project 02-compatible processed interaction table, including drug ID,
   target ID, SMILES, target sequence, and that exact label column.
3. The existing `davis_split_assignments.csv` generated with random state
   `20260830`, covering `random_pair`, `cold_drug`, and `cold_target`.
4. The existing raw Davis files `ligands_can.txt`, `proteins.txt`, and `Y`, and
   their SHA-256 provenance records.

Every required item must be hashed and recorded in the Phase 10 provenance
output. Missing or incompatible items are a fail-closed stop. The runner must
not download data, rebuild a table, regenerate splits, or infer assignments.

## Explicit exclusions

The following Project 10 v0.2 resources are **not** eligible for this Phase 10
analysis:

- its pKd cutoff of 7.0;
- its split seed `20260915`;
- its 20% split assignments or split diagnostics as a replacement for the
  Project 02 split-assignment file;
- any Project 10 v0.2 model predictions, baselines, or outcomes.

## Interpretation

This amendment preserves the original Phase 10 study design. It authorizes a
source-binding preflight only. Experimental execution remains unavailable until
the preflight verifies every frozen Project 02-compatible input.
