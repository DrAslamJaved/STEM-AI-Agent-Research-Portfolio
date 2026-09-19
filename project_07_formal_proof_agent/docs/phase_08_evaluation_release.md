# Phase 08: controlled evaluation and release

## Purpose

Phase 08 closes the study with an auditable comparison of the registered
`llm_only`, `llm_sympy`, and `llm_lean_repair` arms. It aggregates captured
candidate evaluations rather than generating proofs itself. This separation
keeps the release calculation reproducible and prevents an evaluator from
silently changing a theorem or retrying an arm outside its budget.

## Locked design

| Control | Requirement |
| --- | --- |
| Benchmark | miniF2F `v1`, commit `f0dcc8b59e630fba00ba9569ca6714700e0a8801` |
| Lean | `leanprover-community/lean:3.42.1` |
| Mathlib | `cb2b02fff213ed6f65bebd64446baac64137dcda` |
| Development | `valid` split only |
| Final evaluation | Locked `test` split only |
| Formal validity | Independent Lean compilation |
| Lean-repair budget | At most two repair iterations |
| Theorem edits | Forbidden |
| Unsafe shortcuts | Reject `sorry`, `admit`, and `axiom` |

Every arm must use the identical frozen theorem set, model identifier,
sampling budget, and seed schedule. The SymPy arm may use symbolic diagnostics
and exact counterexample search; it receives no Lean feedback. The Lean arm may
receive compiler diagnostics within its bounded repair budget.

## Reported evidence

For each arm, Phase 08 reports compile@1, pass@k, compilation after the allowed
workflow, refutation success, static-policy rejections, elapsed time, token
counts, estimated cost, and recorded human-intervention attempts. It also
reports theorem-paired outcomes and an exact two-sided McNemar p-value for each
tool-assisted arm versus the LLM-only arm. A p-value is descriptive evidence,
not a substitute for the preregistered protocol or Lean validity decision.

The `kind: refutation` records track deliberately false fuzzy-similarity claims
for which a valid counterexample is the intended outcome. Their success rate is
separate from proof compilation and must not be blended into a proof score.

## Release decision

The release bundle is ready only when all required test-split records,
provenance hashes, and policy checks validate and a human reviewer has approved
the evidence. The checked-in synthetic fixture always remains blocked:

- it has `synthetic_demo: true`;
- it records no final miniF2F/model-performance result; and
- it has no human release approval.

The synthetic result therefore demonstrates workflow integrity, not the answer
to the central research question. A future held-out run must retain the raw
attempt records outside the repository where appropriate, commit only the
sanitized manifest/evidence allowed by the data policy, and state all remaining
limitations: benchmark contamination, bounded repair budget, model/version
dependence, cost-time trade-offs, and scope limited to the frozen theorem set.

## Reproduction sequence

Run the deterministic fixture first:

```powershell
python project_07_formal_proof_agent/scripts/run_phase_08_synthetic_release.py `
  --project-root project_07_formal_proof_agent
python project_07_formal_proof_agent/scripts/validate_phase_08_evaluation_release.py
```

For a genuine locked study, place an approved non-synthetic manifest, its
immutable task manifest, and captured attempt JSONL file in a protected results
location, then call `run_phase_08_final_evaluation.py` with all three inputs.
The runner verifies both SHA-256 digests before aggregation and refuses a
synthetic manifest by design.
