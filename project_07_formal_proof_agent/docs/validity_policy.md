# Formal Validity and Safety Policy

## What counts as a valid proof

A candidate is valid only when it:

1. compiles in the pinned Lean environment;
2. satisfies the theorem's stated assumptions;
3. contains no prohibited shortcut;
4. is evaluated independently from the generation workflow.

## Prohibited shortcuts

The evaluator rejects candidates containing:

- `sorry`;
- `admit`;
- unapproved axioms introduced solely to prove the target;
- declarations that bypass the intended theorem assumptions;
- changes to the benchmark theorem statement;
- use of the held-out test split during workflow development.

## Counterexample policy

SymPy, exact arithmetic, and bounded enumeration may refute an overbroad claim.
They do not establish a universal theorem. A counterexample must record the
definition, domain, witness, exact values where possible, and verification method.

## Repair budget

The draft Phase 01 budget is:

- maximum candidates per theorem: 8;
- maximum Lean repair iterations per candidate: 3;
- fixed prompt/model settings within each controlled run.

These values remain subject to human approval before execution.