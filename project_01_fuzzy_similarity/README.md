# Project 01 — Fuzzy Similarity Measures

**AI-Assisted Development and Validation of a Cardinality-Based Fuzzy Similarity Measure**

A reproducible, human-supervised study in which an AI agent and a human
reviewer jointly formulate, analyse, implement, test, and empirically evaluate
a parametric family of fuzzy similarity measures — under a governance protocol
whose single purpose is to keep **conjecture, proof, and numerical evidence
strictly separate**.

The scientifically interesting outcome of this project is an honest one: the
proposed measure is *proven* to satisfy several core properties, and is
*experimentally shown* to offer **no comparative advantage** over classical
baselines on the datasets tested. Both the proofs and the negative empirical
results are first-class deliverables here.

---

## 1. What this project is (and is not)

This is a **research-methodology artifact**, not a production library and not an
autonomous agent. The "agent" is a *human-governed workflow*: every substantive
claim passes through a documented approval gate, and the discipline that keeps
the work honest lives in the protocol documents, the claim ledger, and the
phase-by-phase approval traces.

The guiding research question (`docs/research_protocol.md`) is:

> Can a human-supervised AI-agent workflow support the rigorous formulation,
> mathematical analysis, implementation, testing, and empirical evaluation of a
> cardinality-based fuzzy similarity measure **without confusing generated
> conjectures or numerical observations with proven mathematical results?**

Everything in the repository is organised to answer that question defensibly.

## 2. The measure

For a finite non-empty universe `X = {x₁,…,xₙ}` and fuzzy sets `U, V ∈ [0,1]ⁿ`,
with σ-count cardinalities and the residual difference
`|U∖V| = Σᵢ max(μ_U(xᵢ) − μ_V(xᵢ), 0)`, define the four cardinality components

```
δ = |U ∩ V|            (intersection)
γ = |(U ∪ V)ᶜ|         (co-union)
α = max(|U∖V|, |V∖U|)   (larger residual)
β = min(|U∖V|, |V∖U|)   (smaller residual)
```

which decompose the universe: `δ + α + β + γ = n`.

The parametric rational similarity is

```
            a·δ²  + e·α·β + δ·(b·α + c·β + d·γ)
S_θ(U,V) = ───────────────────────────────────────
            a'·δ² + e'·α·β + δ·(b'·α + c'·β + d'·γ)
```

with the explicit boundary rule `S_θ(U,V) = 1` if `U = V` and `0` otherwise
when the denominator is zero. The **approved reflexive subfamily** used for all
analysis requires `a = a'`, `d = d'`, `0 ≤ b ≤ b'`, `0 ≤ c ≤ c'`, `0 ≤ e ≤ e'`.

The rational form and component names were **supplied by the human
investigator**; the residual-difference convention and the zero-denominator
rule were **agent-proposed and human-approved** (see
`docs/phase_04_candidate_formulation.md`). No novelty is claimed.

The implementation also provides min–max fuzzy **Jaccard**, **Dice**, and
**cosine** as explicitly-documented comparison baselines.

## 3. Claim status (the heart of the method)

Every claim carries one of the statuses defined in
`validation/claim_classification_protocol.md`. The current standing:

| Claim | Status |
|---|---|
| Component decomposition `δ+α+β+γ = n` | **PROVED_VERIFIED** |
| Boundedness `0 ≤ S ≤ 1` (approved subfamily) | **PROVED_VERIFIED** |
| Symmetry `S(U,V) = S(V,U)` | **PROVED_VERIFIED** |
| Reflexivity `S(U,U) = 1` (positive denominator) | **PROVED_VERIFIED** |
| Generic strong identity `S(U,V)=1 ⇒ U=V` | **REFUTED** (counterexample) |
| Generic self-complementarity | **REFUTED** (counterexample) |
| Monotonicity / transitivity | **INCONCLUSIVE** |
| Novelty | **INCONCLUSIVE** |

A hard rule is enforced throughout: **`EXPERIMENTALLY_SUPPORTED ⇏
PROVED_VERIFIED`**. Numerical tests and property-based tests can refute or
support a conjecture; they never promote it to a theorem.

## 4. Experimental results (all honest nulls)

| Phase | Study | Pre-registered effect threshold | Result |
|---|---|---|---|
| 08R | Synthetic clustering (ARI) | mean ARI diff ≥ 0.02 | All four methods ARI = 1.000; **no advantage** |
| 09 | Cardinality-preserving robustness | mean ARI diff ≥ 0.02 | θ₁ vs cosine diff 0.0016 (statistically non-zero, below threshold); **no supported advantage** |
| 10 | External Davis DTI | — | **CLOSED — BLOCKED_MISSING_FROZEN_INPUTS** (provenance could not be verified; study not run) |
| 10R | Re-frozen Davis DTI (cold-drug / cold-target / random) | AP diff ≥ 0.02 **and** bootstrap CI lower > 0 | θ₁ ≈ Jaccard across all splits; **no statistically or practically supported advantage** |

The Phase 10R result is a **valid, reproducible negative result**: under the
residual convention on these character-frequency features, θ₁ is numerically
almost identical to min–max Jaccard. Phase 10 was deliberately *closed as
blocked* rather than reconstructed, because its frozen inputs could not be
recovered with verifiable provenance.

## 5. Repository layout

```
project_01_fuzzy_similarity/
├── src/fuzzy_similarity/core.py   Approved measure + baselines (validated, pure-Python)
├── tests/                         Unit, boundary, negative, and Hypothesis property tests
│   ├── test_core.py
│   ├── test_properties.py         ≥200 generated examples per principal property
│   └── test_phase_10r_runner.py
├── validation/
│   ├── exact_audit.py             Exact-rational (fractions) audit of proof cases
│   ├── claim_classification_protocol.md
│   ├── phase_05_proof_drafts.md   Analytic proofs
│   └── phase_05_independent_audit.md
├── docs/                          Phase protocols, formulation, and completion records
├── agent_trace/                   Per-phase human-approval decision records (the audit trail)
├── experiments/                   Pre-registered experiment runners (phases 07–10R)
├── configs/                       Frozen experiment configurations (seeds, dimensions)
├── literature/                    Reference-verification matrix and search log
├── results/                       Archived outputs with SHA-256 provenance
├── pyproject.toml                 Package metadata + permissive dependency ranges
└── constraints.txt                Fully pinned dependency closure for exact reproduction
```

## 6. Install and reproduce

Requires Python ≥ 3.10. From inside `project_01_fuzzy_similarity/`:

```bash
# Reproducible install against the exact frozen dependency closure
python -m pip install -c constraints.txt ".[test]"

# Run the full test suite with 100% branch coverage enforced
python -m pytest tests/ --cov=src/fuzzy_similarity --cov-branch --cov-fail-under=95

# Run the independent exact-rational proof audit (270 ordered pairs)
python validation/exact_audit.py
```

CI (`.github/workflows/project-01-ci.yml`) runs the tests and the exact audit
on Python 3.10–3.14 for every change to this project.

### Reproducing the external DTI study (Phase 10R)

The Phase 10R runner is intentionally provenance-bound and does **not** carry a
machine-specific path. Point it at the frozen Davis source recorded in
`docs/phase_10r_refrozen_external_dti_protocol.md`:

```bash
export PHASE_10R_SOURCE_ROOT=/path/to/frozen/source    # must contain project_10_.../data/raw/davis
python experiments/run_phase_10r.py --execute
```

The runner hashes the raw source, checks it against the retained manifest,
constructs its own table and leakage-controlled splits, and refuses to proceed
if the source does not match. Results are written atomically with SHA-256
provenance.

## 7. Usage example

```python
from fuzzy_similarity import Parameters, rational_similarity, fuzzy_jaccard

u = [1.0, 0.25, 0.0]
v = [0.5, 0.75, 0.0]

theta = Parameters(
    a=1.0, b=0.25, c=0.5, d=0.75, e=0.2,
    a_prime=1.0, b_prime=0.5, c_prime=0.75, d_prime=0.75, e_prime=0.8,
)
print(rational_similarity(u, v, theta))   # approved reflexive subfamily
print(fuzzy_jaccard(u, v))                 # min–max fuzzy Jaccard baseline
```

All inputs are validated: memberships must be finite and in `[0,1]`, vectors
must have equal length, and `Parameters` enforces the approved subfamily
contract at construction.

## 8. Human-intervention record

The protocol requires at least five meaningful human interventions. Those on
record include: distinguishing reflexivity from identity; forbidding silent
zero-denominator conventions; supplying a counterexample for an unequal-pair
zero denominator; rejecting an invalid Phase 09 test artifact that had been
saved as code; and closing Phase 10 as blocked on provenance grounds. The full
chain is in `agent_trace/` and `docs/critical_reasoning.md`.

---

*Part of the [STEM AI Agent Research Portfolio](https://github.com/DrAslamJaved/STEM-AI-Agent-Research-Portfolio).
This project is a methodology demonstration; its measures are not claimed to be
novel or state-of-the-art.*
