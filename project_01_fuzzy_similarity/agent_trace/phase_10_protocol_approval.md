# Phase 10 protocol approval

Date: 2026-09-17

## Decision

The human reviewer approved the pre-registered external Davis DTI application
protocol after reading `docs/phase_10_external_dti_protocol.md`.

## Authorized scope

Implement the provenance gate and then the external-data runner exactly as
specified.  The provenance gate may inspect existing local Project 02 files and
write a file inventory; it must not generate model outcomes.  Experimental
evaluation requires an explicit `--execute` command after the data schema is
bound and reviewed.
