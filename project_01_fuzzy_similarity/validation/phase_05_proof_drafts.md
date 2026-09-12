# Phase 05 — Analytic proof drafts and counterexamples

## Status discipline

The positive proofs below have completed human review and independent audit. Component decomposition, boundedness, symmetry, and reflexivity are now `PROVED_VERIFIED` under the stated assumptions. The counterexamples are exact analytic constructions and were initially held as `INCONCLUSIVE` pending human review.

**Human review decision:** Dr. Aslam Javed approved the Phase 05 proof-draft review on 2026-09-05T13:08:54Z. The positive results remain `PROOF_DRAFTED` pending independent mathematical verification. The two counterexamples below are accepted as valid refutations of the corresponding generic claims.

**Promotion decision:** Dr. Aslam Javed approved promotion after the independent audit on 2026-09-05T13:55:11Z.

No numerical experiment, symbolic software calculation, or randomized test is used as a theorem proof in this document.

## Assumptions

Use the Phase 04 definitions and the approved full-domain boundary rule. Let \(\Theta_R\) be the human-approved reflexive parameter subfamily:

\[
a=a',\qquad d=d',\qquad
0\le b\le b',\quad 0\le c\le c',\quad 0\le e\le e'.
\]

All parameters are finite and non-negative. Write \(N=N_\theta(U,V)\) and \(D=D_\theta(U,V)\).

## Lemma 1 — Component decomposition

**Statement.** For every \(U,V\in\mathcal F(X)\),

\[
\delta_{U,V}+\alpha_{U,V}+\beta_{U,V}+\gamma_{U,V}=n.
\]

**Proof draft.** At one coordinate, put \(u=\mu_U(x_i)\) and \(v=\mu_V(x_i)\). The residual differences are \((u-v)_+\) and \((v-u)_+\). Hence

\[
\min(u,v)+(u-v)_+ +(v-u)_+=\max(u,v).
\]

Adding \(1-\max(u,v)\) gives 1. Summation over \(i\) gives

\[
\delta+p+q+\gamma=n.
\]

Because \(\alpha+\beta=\max(p,q)+\min(p,q)=p+q\), the stated identity follows. \(\square\)

**Status:** `PROVED_VERIFIED`.

## Proposition 1 — Boundedness

**Statement.** For every \(\theta\in\Theta_R\) and \(U,V\in\mathcal F(X)\),

\[
0\le S_\theta(U,V)\le1.
\]

**Proof draft.** On \(\Theta_R\),

\[
D-N=(b'-b)\delta\alpha+(c'-c)\delta\beta+(e'-e)\alpha\beta\ge0.
\]

All components and coefficients are non-negative, so \(0\le N\le D\) whenever \(D>0\). Thus \(0\le N/D\le1\). If \(D=0\), the approved boundary rule returns 1 for \(U=V\) and 0 otherwise. Both are in \([0,1]\). \(\square\)

**Status:** `PROVED_VERIFIED`.

## Proposition 2 — Symmetry

**Statement.** For every \(\theta\in\Theta_R\),

\[
S_\theta(U,V)=S_\theta(V,U).
\]

**Proof draft.** Swapping \(U\) and \(V\) exchanges \(p\) and \(q\), leaving their maximum \(\alpha\) and minimum \(\beta\) unchanged. Intersection and union are commutative, so \(\delta\) and \(\gamma\) are unchanged. Hence both \(N\) and \(D\), including the zero-denominator branch condition, are unchanged. \(\square\)

**Status:** `PROVED_VERIFIED`.

## Proposition 3 — Reflexivity

**Statement.** For every \(\theta\in\Theta_R\),

\[
S_\theta(U,U)=1.
\]

**Proof draft.** When \(U=V\), \(p=q=\alpha=\beta=0\). Therefore

\[
N=a\delta^2+d\delta\gamma=D,
\]

because \(a=a'\) and \(d=d'\). If \(D>0\), the ratio is 1. If \(D=0\), the approved equal-pair branch also assigns 1. \(\square\)

**Status:** `PROVED_VERIFIED`.

## Strong identity: generic claim is not yet valid

The implication \(S_\theta(U,V)=1\Rightarrow U=V\) does not follow from reflexivity.

### Counterexample draft

Take \(X=\{x\}\), \(U=(1)\), \(V=(1/2)\), and admissible parameters

\[
a=a'=1,\quad b=b'=1,\quad c=c'=d=d'=e=e'=0.
\]

Then \(\delta=1/2\), \(\alpha=1/2\), \(\beta=\gamma=0\), and

\[
N=D=\tfrac12,qquad S_\theta(U,V)=1,qquad U\ne V.
\]

Thus the generic strong-identity claim is a proposed refutation, not a valid theorem of \(\Theta_R\).

**Status of generic identity claim:** `REFUTED` by the human-approved counterexample above.

### Agent-proposed sufficient condition

For a possible identity-preserving subfamily, add

\[
b<b'\quad\text{and}\quad (e'=0\ \text{or}\ e<e').
\]

**Proof sketch.** If \(U\ne V\), then \(\alpha>0\). If \(\delta>0\), strict \(b<b'\) gives \(D-N>0\). If \(\delta=0\) and \(\alpha\beta=0\), then \(D=0\) and the unequal-pair branch gives 0. If \(\delta=0\) and \(\alpha\beta>0\), either \(e'=0\), which forces \(e=0\) and again gives the unequal-pair branch, or \(e<e'\), which gives \(D-N>0\). In each case \(S_\theta(U,V)<1\). The converse \(U=V\Rightarrow S_\theta(U,V)=1\) is Proposition 3. \(\square\)

**Status:** `AGENT_PROPOSED` condition with `PROOF_DRAFTED` argument; not yet approved.

## Self-complementarity: generic claim is not yet valid

For the baseline control parameters

\[
a=a'=1,\quad b=c=d=e=0,\quad b'=c'=1,\quad d'=e'=0,
\]

the measure is the min-max fuzzy Jaccard control. Let \(X=\{x_1,x_2\}\), \(U=(1,0)\), and \(V=(1,1)\). Then

\[
S(U,V)=\tfrac12.
\]

For complements, \(U^c=(0,1)\) and \(V^c=(0,0)\); their numerator and denominator are both zero, and the unequal-pair branch gives

\[
S(U^c,V^c)=0.
\]

Therefore \(S(U,V)\ne S(U^c,V^c)\).

**Status of generic self-complementarity claim:** `REFUTED` by the human-approved counterexample above.

## Monotonicity and transitivity

No universal claim is stated at this stage. The chosen fuzzy-set order, the sets allowed to vary, and the inequality direction must be fixed before either property can be analysed. Their status is `INCONCLUSIVE`.

## Review checklist

1. Check residual-difference algebra in Lemma 1.
2. Check the factorization of \(D-N\) in Proposition 1.
3. Check all zero-denominator branches in Propositions 1–3.
4. Both counterexamples were reviewed and accepted by the human researcher.
5. Independent audit completed and promotion approved; preserve its reproducibility artifacts.
6. Decide whether to approve the proposed sufficient identity conditions before implementation.
