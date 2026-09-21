# Phase 09: real benchmark execution

## Purpose

Phases 01–08 built and tested the Project 07 research infrastructure.  Their
committed outputs are synthetic workflow fixtures.  Phase 09 is the first
phase that can answer the research question with real, model-generated Lean
candidates:

> Under a fixed model, theorem set, seed schedule, sampling budget, and cost
> cap, does SymPy or bounded Lean repair improve independently Lean-compiled
> proof completion relative to LLM-only generation?

## Canonical deployment

The only canonical source location is
`G:\Research\STEM\Micro1_STEM_Agent_Portfolio\project_07_formal_proof_agent`.
The Phase 09 provisioning script creates the ignored miniF2F checkout under
that project directory.  It does not move or delete the historical Phase 02
worktree, so failure in the new canonical runtime cannot destroy the working
toolchain already available elsewhere.

The repository also forces Project 07 text to LF on checkout and computes the
Phase 07 Lean-source digest from canonical LF-normalized text.  This preserves
the provenance link across Windows CRLF worktrees without treating line-ending
conversion as a change to the formal statement.

## Frozen task construction

The task preparer verifies miniF2F commit
`f0dcc8b59e630fba00ba9569ca6714700e0a8801` and extracts only the declaration
preceding each `begin` proof.  It rejects any source whose parser does not
recover exactly 244 theorems per split.  Therefore model prompts do not receive
the existing Lean proof bodies from miniF2F files.

The fixed four-item fuzzy suite contains three proof tasks and one refutation
task.  Its custom preamble contains only definitions—never the previously
proved Phase 07 theorems—so a candidate must establish the requested property
under independent Lean compilation.  The refutation records the exact
quarter-versus-half singleton Jaccard witness separately from proof scores.

## Controlled arms

| Arm | During generation | Lean evaluation |
| --- | --- | --- |
| `llm_only` | Frozen theorem only; no tool or feedback | After the full candidate batch closes |
| `llm_sympy` | Frozen theorem plus recorded exact SymPy context only | After the full candidate batch closes |
| `llm_lean_repair` | Frozen theorem; then at most two compiler-informed repairs | Each attempt, because diagnostics are the permitted feedback |

All arms use identical tasks, initial sample indices, model identifier, and
initial seed schedule.  The live runner rejects a provider response whose
reported model identifier differs from the preregistered one.

## Evidence and release boundary

Every attempt records the prompt, raw completion, candidate source hash,
independent Lean result, static-policy result, provider response id, token
usage, estimated cost, and generation/compilation times.  API keys are never
written to the evidence.

Each live run also appends a LF-stable `generation_journal.jsonl` before the
candidate is evaluated.  If a network failure or cost-cap stop interrupts a
run, paid completions already received remain auditable rather than vanishing
from the local evidence directory.

A `valid` split run yields development evidence only.  A `test` split run
requires explicit confirmation, produces a hashed Phase 08 evidence manifest,
and starts with `human_review.approved: false`.  A final claim is therefore
blocked until a human reviews the raw records and limitations.

The approval command requires both `--approve` and a substantive review note.
It rechecks the raw-candidate SHA-256 before changing the release bundle; a
changed candidate file invalidates approval rather than being silently reused.

No conclusion should claim general mathematical intelligence.  Even a fully
reviewed run is conditional on the frozen miniF2F set, the model/version,
prompting policy, bounded repair budget, test contamination risk, and the
recorded cost-time trade-off.
