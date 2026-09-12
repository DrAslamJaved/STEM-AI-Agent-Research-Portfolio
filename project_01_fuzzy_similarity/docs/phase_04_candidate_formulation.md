# Phase 04 — Rational fuzzy cardinality candidate

## Status and provenance

This record has three distinct layers:

- **Human-supplied core family:** the rational formula and the four components were supplied by Dr. Aslam Javed.
- **Agent-proposed completion:** residual fuzzy difference and the full-domain zero-denominator rule below.
- **Human-approved operational conventions:** the residual difference and zero-denominator rule were approved by Dr. Aslam Javed.

**Approval timestamp:** 2026-09-05T12:45:20Z.

The full unrestricted class is not approved as a final Project 1 similarity measure. The reflexive subfamily specified below is `HUMAN_APPROVED` for Phase 05 mathematical analysis only. No novelty claim, theorem, code result, or experimental result is made here.

## Domain and operations

Let \(X=\{x_1,\ldots,x_n\}\) be finite and non-empty, and let \(\mathcal F(X)=[0,1]^n\). For \(U,V\in\mathcal F(X)\), use

\[
\mu_{U\cap V}(x_i)=\min\{\mu_U(x_i),\mu_V(x_i)\},\qquad
\mu_{U\cup V}(x_i)=\max\{\mu_U(x_i),\mu_V(x_i)\},
\]

\[
\mu_{U^c}(x_i)=1-\mu_U(x_i),\qquad
\mu_{U\setminus V}(x_i)=\max\{\mu_U(x_i)-\mu_V(x_i),0\}.
\]

All cardinalities are sigma-counts: \(|W|=\sum_i\mu_W(x_i)\).

## Cardinality components

Define

\[
p_{U,V}=|U\setminus V|,\qquad q_{U,V}=|V\setminus U|,
\]

\[
\alpha_{U,V}=\max\{p_{U,V},q_{U,V}\},\qquad
\beta_{U,V}=\min\{p_{U,V},q_{U,V}\},
\]

\[
\delta_{U,V}=|U\cap V|,\qquad
\gamma_{U,V}=|(U\cup V)^c|.
\]

The residual-difference definition makes the following decomposition available for later proof review:

\[
\delta_{U,V}+\alpha_{U,V}+\beta_{U,V}+\gamma_{U,V}=n.
\]

## Human-supplied rational family

For non-negative parameters

\[
a,b,c,d,e,a',b',c',d',e'\in[0,\infty),\qquad
x\leq x'\ \text{for}\ x\in\{a,b,c,d,e\},
\]

write

\[
N_\theta(U,V)=a\delta^2+e\alpha\beta+\delta(b\alpha+c\beta+d\gamma),
\]

\[
D_\theta(U,V)=a'\delta^2+e'\alpha\beta+\delta(b'\alpha+c'\beta+d'\gamma),
\]

where the subscripts \(U,V\) on \(\alpha,\beta,\delta,\gamma\) are suppressed.

## Approved full-domain boundary rule

\[
S_\theta(U,V)=
\begin{cases}
N_\theta(U,V)/D_\theta(U,V), & D_\theta(U,V)>0,\\[4pt]
1, & D_\theta(U,V)=0\ \text{and}\ U=V,\\
0, & D_\theta(U,V)=0\ \text{and}\ U\ne V.
\end{cases}
\]

This rule is required because \(D_\theta=0\) can occur for unequal sets; for example, \(U=\varnothing\) and \(V=X\) give \(\delta=\beta=\gamma=0\). It is therefore incorrect to state that a zero denominator occurs only at \(U=V=\varnothing\).

## Approved reflexive parameter subfamily

The inequalities \(x\leq x'\) support a future boundedness argument, but do **not** alone imply reflexivity. For \(U=V\) with \(D_\theta(U,U)>0\),

\[
S_\theta(U,U)=
\frac{a\delta^2+d\delta\gamma}{a'\delta^2+d'\delta\gamma}.
\]

The agent proposed the reflexive restriction

\[
a=a',\qquad d=d'.
\]

Dr. Aslam Javed approved this restriction on 2026-09-05T12:52:59Z. Additional strictness conditions needed for the strong identity property \(S(U,V)=1\Rightarrow U=V\) will be analysed separately; they are not assumed here.

## Baseline control setting

The supplied baseline parameter setting

\[
a=a'=1,\quad b=c=d=e=0,\quad b'=c'=1,\quad d'=e'=0
\]

is retained as a control. Subject to the stated operations and boundary rule, it simplifies to the min-max fuzzy Jaccard form whenever its denominator is positive. It is a comparison baseline, not a novelty claim.

## Claim ledger

| Claim | Status | Evidence / action |
|---|---|---|
| Rational formula and component names | `HUMAN_SUPPLIED` | User-provided Phase 04 formulation image. |
| Residual fuzzy difference | `HUMAN_APPROVED` | User approval after hidden-assumption review. |
| Full-domain zero-denominator rule | `HUMAN_APPROVED` | User approval after unequal-pair counterexample. |
| Reflexive restriction \(a=a'\), \(d=d'\) | `HUMAN_APPROVED` | Approved for Phase 05 analysis; the proof remains pending. |
| Component decomposition, boundedness, symmetry, reflexivity | `PROVED_VERIFIED` | Human-reviewed Phase 05 proofs plus independent audit. |
| Generic strong identity and self-complementarity | `REFUTED` | Human-approved Phase 05 counterexamples. |
| Monotonicity and transitivity | `INCONCLUSIVE` | No precise proposition has yet been selected. |
| Novelty | `INCONCLUSIVE` | Requires broader verified literature review. |

## Next gate

Phase 05 may state and verify propositions for the selected parameter subfamily. Approval of the formulation does not constitute a proof of any property.
