# Phase 06 — Lean Generate–Verify–Repair Prompts

## Initial attempt

```text
Produce only the Lean proof body for the frozen theorem below.
Do not restate, edit, or weaken the theorem declaration.
This is the initial attempt; no compiler feedback is available yet.
Do not use sorry, admit, or axiom.

{frozen_task_imports_and_declaration}
```

## Bounded repair attempt

```text
Produce only a replacement Lean proof body for the frozen theorem below.
Do not restate, edit, or weaken the theorem declaration.
Use the independent Lean feedback to repair the previous proof body.
Do not use sorry, admit, or axiom.

{previous_proof_body_and_independent_lean_feedback}

Frozen theorem (Lean repair iteration {iteration}):
{frozen_task_imports_and_declaration}
```

## Protocol constraints

- Initial generation gets only the frozen theorem.
- A repair request may contain the immediately preceding proof body and its
  static-policy result or Lean stdout/stderr from the independent evaluator.
- The theorem declaration, imports, benchmark record, and split cannot change.
- Stop when a candidate compiles or the pre-registered repair budget is spent.
- Preserve every prompt, raw completion, proof-body digest, compiler result,
  and feedback digest before reporting metrics.
- Lean compilation—not a model claim or SymPy result—decides formal validity.
