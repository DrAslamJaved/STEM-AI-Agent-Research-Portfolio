# Phase 05 — Independent mathematical audit

## Scope and status

This audit independently rechecks the positive Phase 05 proof drafts under the approved assumptions

\[
a=a',\qquad d=d',\qquad 0\le b\le b',\quad 0\le c\le c',\quad 0\le e\le e'.
\]

It consists of a direct analytic re-derivation and an exact-rational finite-grid check. The grid check is reproducible audit evidence only; it does not replace the analytic arguments.

The positive claims remain `PROOF_DRAFTED` until Dr. Aslam Javed reviews this audit and authorizes promotion.

## Independent analytic re-derivation

### Component decomposition

For scalars \(u,v\in[0,1]\), exactly one of \((u-v)_+\) and \((v-u)_+\) is nonzero. Therefore

\[
\min(u,v)+(u-v)_++(v-u)_+=\max(u,v).
\]

Adding \(1-\max(u,v)\) produces 1 at each coordinate. Summing, then using \(\alpha+\beta=p+q\), yields \(\delta+\alpha+\beta+\gamma=n\).

### Boundedness

Substitution of \(a'=a\) and \(d'=d\) gives the exact identity

\[
D-N=(b'-b)\delta\alpha+(c'-c)\delta\beta+(e'-e)\alpha\beta.
\]

Every factor is non-negative, hence \(N\le D\) whenever \(D>0\). The separately defined \(D=0\) branches are 0 and 1, so they also lie in \([0,1]\).

### Symmetry

Exchanging \(U,V\) exchanges the directed residual cardinalities \(p,q\) but leaves their maximum and minimum unchanged. It also leaves intersection and union cardinalities unchanged. Thus the complete branch definition is symmetric.

### Reflexivity

For \(U=V\), \(\alpha=\beta=0\). The two expressions reduce to the common value \(a\delta^2+d\delta\gamma\). A positive common denominator produces quotient 1; otherwise the equal-pair boundary rule produces 1.

## Exact-rational computational cross-check

Executed command:

```text
python3 project_01_fuzzy_similarity/validation/exact_audit.py
```

Result:

```json
{"audit":"exact_rational_grid","checked_pairs":270,"membership_values":["0","1/2","1"],"parameter_sets":3,"result":"passed"}
```

The audit used exact `fractions.Fraction` arithmetic, not floating point. It enumerated all ordered pairs of fuzzy membership vectors of dimensions 1 and 2 over \(\{0,1/2,1\}\), for three admissible parameter settings. It checked the component decomposition, boundedness, symmetry, reflexivity, and both accepted counterexamples.

Script SHA-256:

```text
a89e2ecfb56648e63e5d112843d7700a4c2ae0ce92d219cdb87bb4717913bdc4
```

## Audit conclusion

The independent derivation agrees with all positive proof drafts, and the exact-grid audit found no contradiction in its finite scope. This is sufficient evidence to request human promotion review; it is not, by itself, a new theorem proof.

## Next gate

Dr. Aslam Javed may now decide whether to promote component decomposition, boundedness, symmetry, and reflexivity from `PROOF_DRAFTED` to `PROVED_VERIFIED`.
