# Phase 06 — Implementation and Test Contract

## Status

`HUMAN_APPROVED` to implement the Phase 04/05 reflexive subfamily. This
document records the software contract. It does not add a similarity formula,
novelty claim, or theorem.

## Implemented formulation

The implementation accepts finite, non-empty equal-length membership vectors
in `[0, 1]`. It uses sigma-count cardinality, minimum intersection, maximum
union, and the approved residual difference:

\[
\mu_{U\setminus V}(x)=\max(\mu_U(x)-\mu_V(x),0).
\]

It computes \(\alpha,\beta,\delta,\gamma\) as approved in Phase 04 and the
rational formula with the full-domain zero-denominator rule:

\[
D=0 \Rightarrow S(U,V)=\begin{cases}1&U=V\\0&U\ne V.\end{cases}
\]

Only the approved Phase 05 subfamily is accepted:

\[
a=a',\quad d=d',\quad 0\le b\le b',\quad 0\le c\le c',\quad 0\le e\le e'.
\]

## Validation scope

The initial suite includes unit, boundary, negative-input, fixed-seed
randomized property, numerical-sanity, and regression-counterexample tests.
The Phase 05 exact rational audit remains the independent mathematical audit.
Passing implementation tests verifies code behaviour on their stated cases;
it does **not** prove a mathematical theorem.

## Control baseline

`Parameters.jaccard_control()` encodes the approved Phase 04 min-max fuzzy
Jaccard control setting. It is a baseline, not a claim that Project 1 has
introduced a new measure.

## Commands

From `project_01_fuzzy_similarity`:

```powershell
python -m unittest discover -s tests -v
python validation/exact_audit.py
```

The project intentionally has no mandatory third-party testing dependency in
this phase.
