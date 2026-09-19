# Phase 07 - Fuzzy Similarity Formalization Suite

## Corrected portable MVP

The first compiler run exposed two Lean 3 / Windows compatibility defects:
imports must precede every module comment, and Unicode mathematical notation
was decoded incorrectly by the pinned Windows Lean executable.  This revision
removes both hazards.  The Lean source is ASCII-only and has no imports.

The mathematical scope is a one-element finite universe with an exact fixed
membership grid:

`{0, 1/4, 1/2, 3/4, 1}`.

Each membership and similarity is represented by an exact symbolic constructor.
The complete similarity range induced by this grid is
`{0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}`.  This avoids floating-point arithmetic and
does not round values such as `1/3` or `2/3`.

## Formal object

For registered grid memberships `a` and `b`, `fuzzy_jaccard` implements the
exact singleton Jaccard table:

- `J(0,0)` is exactly `1` (the explicit empty-union convention).
- Every other finite-grid pair has its exact rational Jaccard value.

The custom source is
[`formalizations/lean3/FuzzySimilarity.lean`](../formalizations/lean3/FuzzySimilarity.lean).

## Lean claims

| Category | Lean theorem | Meaning |
| --- | --- | --- |
| Valid | `fuzzy_jaccard_zero_zero` | The zero-union convention holds. |
| Valid | `fuzzy_jaccard_refl` | Every registered membership is reflexive. |
| Valid | `fuzzy_jaccard_symm` | The grid relation is symmetric. |
| Witness | `quarter_has_one_unit`, `half_has_two_units` | The witness is exactly `1/4` and `1/2`. |
| Counterexample value | `fuzzy_jaccard_counterexample_value` | `J(1/4,1/2)` is exactly `1/2`. |
| Counterexample | `fuzzy_jaccard_counterexample_not_one` | That value cannot equal similarity `1`. |

## Validity boundary

The manifest records source provenance only.  Formal validity requires the
unchanged source to compile with the Phase 02 pinned command:

`elan run leanprover-community/lean:3.42.1 lean`

from the configured miniF2F v1 checkout.  Only evidence with
`status: compiled`, `compilation_passed: true`, and exit code `0` supports a
formal-proof claim.  Never stage a failed compile record.

## Scope

This is deliberately a portable finite-grid MVP.  It does not yet formalize
arbitrary rational memberships, general finite universes, Dice/cosine
similarities, t-norm transitivity, or a miniF2F/model-performance score.  It
establishes a sound compiler-validated basis for those extensions.
