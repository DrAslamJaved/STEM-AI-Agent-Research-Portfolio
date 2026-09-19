# Phase 05 — LLM + SymPy Proof Prompt

Use this immutable template for the SymPy-assisted arm. Insert only a frozen
task declaration and provenance-preserving SymPy context. The model returns a
Lean proof body and never sees Lean compiler diagnostics or an earlier proof.

```text
Produce only the Lean proof body for the frozen theorem below.
Do not restate, edit, or weaken the theorem declaration.
You may use only the SymPy context below; you will receive no Lean compiler
diagnostics, compilation status, search results, or prior candidate text.
SymPy evidence is advisory and cannot establish formal validity.
Do not use sorry, admit, or axiom.

{sympy_exact_diagnostics_and_counterexamples}

Frozen theorem (SymPy repair iteration {iteration}):
{frozen_task_imports_and_declaration}
```

## Protocol constraints

- SymPy may simplify an exact symbolic residual and enumerate a bounded grid of
  rational fuzzy-membership values to find a counterexample.
- Do not treat a no-witness result as proof outside the declared finite grid.
- Do not use floating-point evidence for a counterexample claim.
- Do not pass Lean errors, compilation status, prover-search output, previous
  candidate text, retrieved proofs, or `test`-split tasks to the provider.
- Capture full prompt/completion provenance and SymPy evidence before the
  independent Phase 03 evaluator is invoked.
- Lean compilation remains the sole formal-validity decision.
