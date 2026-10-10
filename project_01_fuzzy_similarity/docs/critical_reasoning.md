# Critical Reasoning Record

This document consolidates the **critical-reasoning interventions** that shaped
Project 1 — the points where a human reviewer challenged, constrained, or
corrected the AI agent's output. It is the narrative companion to the
per-phase approval traces in `agent_trace/` and to the structured claim ledger
in `docs/phase_04_candidate_formulation.md`.

Its purpose is to make the project's central discipline auditable: that no
conjecture was mistaken for a proof, no numerical result for a theorem, and no
unverified source for established theory. The research protocol
(`docs/research_protocol.md`) requires at least five meaningful interventions;
the record below documents more than that, each with the exact claim affected
and the decision reached.

---

## Intervention 1 — Reflexivity is not identity

**Phase:** 01 (mathematical contract).
**Trigger:** the natural temptation to treat `S(A,A) = 1` as if it also
established the strong identity property `S(A,B) = 1 ⇒ A = B`.
**Correction.** The contract was required to state explicitly that

> `S(A,A) = 1 ⇏ [S(A,B) = 1 ⇔ A = B]`.

Reflexivity and identity were thereafter analysed as **separate** properties.
This single distinction is what later allowed generic identity to be *refuted*
while reflexivity was *proven* — two outcomes that a careless treatment would
have conflated.
**Status effect:** identity held open, not assumed.

## Intervention 2 — No silent zero-denominator convention

**Phase:** 01 → 04.
**Trigger:** the rational form `N_θ / D_θ` is undefined when `D_θ = 0`, and it
is tempting to paper over this with an implicit convention.
**Correction.** Silent conventions were forbidden. The definition was required
to carry an explicit, total boundary rule:

> `S_θ(U,V) = 1` if `U = V`, and `0` otherwise, whenever `D_θ(U,V) = 0`.

**Status effect:** the measure is well-defined on the whole domain, and the
boundary branch is independently tested.

## Intervention 3 — A zero denominator can occur for *unequal* sets

**Phase:** 04 (candidate formulation).
**Trigger:** an initial, plausible-sounding assumption that the denominator
vanishes only at `U = V = ∅`.
**Correction.** A counterexample was required and supplied: `U = ∅`, `V = X`
gives `δ = β = γ = 0`, so `D_θ = 0` with `U ≠ V`. It is therefore *incorrect*
to claim the zero denominator occurs only at the empty-set pair, and the
boundary rule (Intervention 2) must handle the unequal case — which it does.
**Status effect:** the full-domain boundary rule was approved *because of* this
counterexample, not in spite of it.

## Intervention 4 — Coefficient dominance does not by itself give reflexivity

**Phase:** 04 → 05.
**Trigger:** the inequalities `b ≤ b'`, `c ≤ c'`, `e ≤ e'` support a
boundedness argument, and it is tempting to assume they also deliver
reflexivity.
**Correction.** They do not. For `U = V` with positive denominator,
`S_θ(U,U) = (a·δ² + d·δ·γ)/(a'·δ² + d'·δ·γ)`, which equals 1 only under the
additional restriction `a = a'`, `d = d'`. That reflexive restriction was
stated separately and approved only for analysis.
**Status effect:** the approved subfamily is `a = a'`, `d = d'` with
coefficient dominance on `b, c, e`; reflexivity is proven *within* it, not
assumed.

## Intervention 5 — Proofs may not be promoted on agent confidence

**Phase:** 05 (proof review and promotion).
**Trigger:** the agent produced proof drafts for decomposition, boundedness,
symmetry, and reflexivity.
**Correction.** Drafts were held at `PROOF_DRAFTED` and promoted to
`PROVED_VERIFIED` only after (a) human review of every logical step and (b) an
**independent exact-rational audit** (`validation/exact_audit.py`) over a
finite grid using exact `fractions` — never floating point — to rule out
round-off artefacts. The two over-general claims (strong identity,
self-complementarity) were `REFUTED` by exact analytic counterexamples.
**Status effect:** four properties `PROVED_VERIFIED`; two `REFUTED`; promotion
was by *added evidence*, never by assertion.

## Intervention 6 — Literature stays `INCONCLUSIVE` until verified at the source

**Phase:** 02 → 03.
**Trigger:** bibliographic metadata for the cardinality-based similarity
literature was readily available and could have been labelled "established".
**Correction.** Metadata agreement is not source verification. Of the recorded
references, only the claims verified against a primary-source page image (the
Zadeh 1965 fuzzy-set definition, p. 339) were marked `ESTABLISHED`; the
remainder — including the De Baets 2001 rational-cardinality family from which
this formulation descends — remain `INCONCLUSIVE` pending full-text
verification. Novelty is therefore also `INCONCLUSIVE`, not claimed.
**Status effect:** the verified literature foundation is deliberately small and
honestly bounded.

## Intervention 7 — A test artifact saved as code must fail, not be rescued

**Phase:** 09 (property testing).
**Trigger:** the initial property-test prompt was accidentally saved into the
test file as Python, and pytest rejected it with a `SyntaxError` during
collection.
**Correction.** The invalid artifact was identified and rejected rather than
patched into something that merely passed. Valid Hypothesis tests (≥200
examples per principal property) were then supplied, raising coverage from
88.8% to 100%. A standing rule was enforced: **the implementation is never
modified merely to make a generated test pass**, and any counterexample
Hypothesis finds is preserved and reported, not suppressed.
**Status effect:** 100% statement and branch coverage, earned honestly.

## Intervention 8 — A study blocked on provenance is closed, not reconstructed

**Phase:** 10 → 10R.
**Trigger:** the pre-registered external Davis DTI study required specific
frozen interaction and split files that could not be recovered with verifiable
provenance.
**Correction.** Phase 10 was **CLOSED — BLOCKED_MISSING_FROZEN_INPUTS**. The
missing inputs were explicitly *not* reconstructed, resplit, re-downloaded, or
substituted under the Phase 10 identifier. A separate, re-frozen study
(Phase 10R) was defined with its own source binding, construction procedure,
split seed, and execution gate, and its results were never allowed to be
described as Phase 10 results.
**Status effect:** a provenance finding was recorded as such — not disguised as
a performance result.

## Intervention 9 — Report the negative result; do not launder it into equivalence

**Phase:** 08R, 09, 10R (interpretation).
**Trigger:** across the clustering, robustness, and external-DTI studies, the
proposed θ₁ measure showed no advantage over Jaccard, Dice, or cosine.
**Correction.** Each study was held to a **pre-registered** practical-effect
threshold (mean ARI difference ≥ 0.02; or AP difference ≥ 0.02 *and* a bootstrap
CI lower bound above zero). Differences that were statistically non-zero but
below the threshold (θ₁ vs cosine at ε = 0.20 in Phase 09) were recorded as
**no supported advantage**. Equally, the absence of a difference was explicitly
*not* claimed as evidence of equivalence.
**Status effect:** valid, reproducible negative results — reported honestly as
the project's empirical outcome.

---

## Summary

| # | Intervention | Claim affected | Outcome |
|---|---|---|---|
| 1 | Reflexivity ≠ identity | identity | held open → later `REFUTED` |
| 2 | No silent zero-denominator rule | definition totality | explicit boundary rule |
| 3 | Zero denominator at unequal sets | denominator domain | counterexample → rule approved |
| 4 | Dominance ⇏ reflexivity | reflexive subfamily | restriction `a=a'`, `d=d'` |
| 5 | No promotion on confidence | 4 properties + 2 refutations | `PROVED_VERIFIED` / `REFUTED` |
| 6 | Verify literature at source | novelty, references | `INCONCLUSIVE` where unverified |
| 7 | Failing test not rescued | implementation correctness | 100% coverage, honest |
| 8 | Block on provenance | Phase 10 external study | closed, not reconstructed |
| 9 | Report the null | comparative advantage | reproducible negative result |

Every final conclusion in this project traces to one or more of these decisions
and preserves its highest *justified* status — no higher.
