# Phase 07 Agent Trace - Fuzzy Similarity Formalization

## Correction recorded

1. The first pinned Lean 3 compilation failed before theorem checking because
   a module comment preceded `import tactic`, and the Windows executable
   misdecoded Unicode source symbols.
2. The corrected source has no imports and is strictly ASCII.
3. The formalization scope is an exact denominator-four membership grid on a
   one-element finite universe, with symbolic exact similarity outputs.

## Decisions locked

1. The grid is `{0, 1/4, 1/2, 3/4, 1}`; all membership and similarity values
   are exact symbolic constructors, including `1/3` and `2/3`.
2. The zero-union convention is explicit: `J(0,0)=1`.
3. Lean proves zero convention, reflexivity, symmetry, witness encodings, and
   the counterexample `J(1/4,1/2)=1/2`.
4. The false universal identity claim is refuted by the compiled theorem that
   the witness does not equal similarity one.
5. The manifest is provenance only; independent pinned Lean compilation is the
   deciding validity event.
6. Failed compiler evidence is not staged, committed, or reported as a proof.

## Phase gate

Phase 07 is ready for review only after the corrected ASCII Lean source
compiles with Lean 3.42.1 from the pinned miniF2F v1 checkout.  This phase
still produces no miniF2F or model-performance score.
