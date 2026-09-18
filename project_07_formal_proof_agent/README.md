# Project 07 — Formal Mathematical Proof and Counterexample Agent

## Research question

Does a controlled generate–verify–repair workflow using SymPy and Lean produce
a higher rate of formally valid mathematical proofs than LLM-only generation?

## Experimental arms

1. **LLM-only** — generates Lean proof candidates without verification feedback.
2. **LLM + SymPy** — uses symbolic simplification and exact counterexample search,
   but receives no Lean compiler feedback during generation.
3. **LLM + Lean repair** — receives Lean diagnostics and performs bounded repair.

All final proof candidates are evaluated independently by Lean compilation.

## Benchmarks

- Frozen miniF2F v1, with its exact commit and Lean 3 toolchain recorded in Phase 02.
- Custom finite fuzzy-set axiom and similarity-measure suite.

## Scientific boundary

A natural-language proof is not treated as valid merely because it is persuasive.
Formal Lean compilation, assumption checks, and counterexample testing determine
validity. The project remains human-governed at every approval gate.