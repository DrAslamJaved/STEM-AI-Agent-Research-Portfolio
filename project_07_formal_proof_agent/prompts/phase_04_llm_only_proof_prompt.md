# Phase 04 — LLM-Only Proof Prompt

Use this immutable template for every baseline sample. The task declaration is
inserted from the pinned manifest; the model returns only the proof body.

```text
Produce only the Lean proof body for the frozen theorem below.
Do not restate, edit, or weaken the theorem declaration.
You have no tools and will receive no feedback or repair suggestions.
Do not use sorry, admit, or axiom.

{frozen_task_imports_and_declaration}
```

## Protocol constraints

- The generation request contains only the frozen task, prompt version, model
  identifier, sample index, and optional seed.
- Do not provide Lean diagnostics, compilation status, SymPy output,
  counterexamples, retrieved proofs, test results, or previous candidate text.
- Each sample has `arm: llm_only` and `iteration: 0`.
- Capture the full prompt, raw completion, source SHA-256, model identifier,
  seed, and token counts before post-hoc evaluation begins.
- Run the Phase 03 evaluator only after the full generation batch is closed.
- Use `valid` tasks for development. The held-out `test` split remains locked.
