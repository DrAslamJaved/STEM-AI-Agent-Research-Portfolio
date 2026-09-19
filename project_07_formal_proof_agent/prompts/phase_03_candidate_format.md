# Phase 03 Candidate Record Format

Every arm emits the same JSONL candidate record. Generation and verification
are intentionally separate so that the independent evaluator can apply the
same final Lean decision to every arm.

```json
{
  "candidate_id": "stable-attempt-id",
  "theorem_id": "pinned-manifest-theorem-id",
  "arm": "llm_only | llm_sympy | llm_lean_repair",
  "lean_code": "by\n  ...",
  "prompt_id": "immutable-prompt-version",
  "iteration": 0,
  "metadata": {"model": "recorded-model-id", "seed": 0}
}
```

Rules:

- Emit only the proof body in `lean_code`; never restate or alter the theorem
  declaration. The evaluator owns the declaration from the pinned manifest.
- Preserve all source text and metadata. The evaluator records SHA-256 source
  digests, compiler diagnostics, exit code, elapsed time, and final status.
- `llm_only` receives no verification feedback during generation. Its proof is
  still compiled independently after generation.
- `llm_sympy` may use symbolic diagnostics and exact counterexample search, but
  it cannot claim formal validity without the same post-hoc Lean compilation.
- `llm_lean_repair` may use bounded compiler diagnostics during repair; every
  attempt remains a separately captured candidate.
- Do not use `sorry`, `admit`, or `axiom`. The harness statically rejects these
  tokens and will not invoke Lean for that candidate.
- The `test` split is unavailable in development. It may be evaluated only in
  an explicitly selected final-evaluation execution mode.
