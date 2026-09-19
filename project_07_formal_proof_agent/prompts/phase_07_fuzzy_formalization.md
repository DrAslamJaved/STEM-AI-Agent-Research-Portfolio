# Phase 07 - Fuzzy Formalization Prompt Contract

Use this contract only when a provider proposes a Lean proof body for a
separately registered fuzzy-similarity theorem.

```text
Return only the Lean proof body for the immutable theorem declaration below.
Do not change its definition, imports, hypotheses, theorem statement, or the
registered exact-grid witness. Do not use sorry, admit, or axiom.

Use ASCII Lean 3 syntax only. The formalization uses a singleton fuzzy universe
and the exact membership grid {0, 1/4, 1/2, 3/4, 1}. Similarities use exact
symbolic values, including 1/3 and 2/3; do not introduce rounding.

{frozen_lean_definition_and_theorem_declaration}
```

The proposed proof is only a candidate.  The Phase 07 compiler runner must
compile the unchanged source under Lean 3.42.1 in the pinned miniF2F v1
environment before any theorem is reported as formally valid.  For a false
claim, retain the exact witness and compile its refutation; do not replace it
with floating-point testing.
