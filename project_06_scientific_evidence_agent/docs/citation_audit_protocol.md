# Phase 06 citation-audit protocol

## Purpose

Select an abstention and citation-acceptance policy without tuning it on the
ordinary SciFact development set. The policy acts on frozen verifier outputs;
it does not introduce a new language model or access gold fields at runtime.

## Split correction

The supplied SciFact five-fold directory partitions 1,109 claims: 809 ordinary
training claims plus 300 ordinary development claims. A direct use of
`claims_train_i.jsonl` would therefore leak ordinary-development labels into
selection. For fold *i*, this project instead defines:

\[
V_i = \operatorname{ids}(\texttt{claims\_dev\_i}) \cap
      \operatorname{ids}(\texttt{claims\_train}),
\qquad
T_i = \operatorname{ids}(\texttt{claims\_train}) \setminus V_i.
\]

The ordinary development IDs are excluded from every \(T_i\) and \(V_i\).
The supplied `claims_train_i.jsonl` files are deliberately not used.

## Fixed policy grid and selection rule

Each candidate policy contains:

- assertion threshold in `{0.45, 0.55, 0.65, 0.75, 0.85, 0.95}`;
- sentence-evidence threshold in `{0.50, 0.60, 0.70, 0.80, 0.90}`; and
- maximum citation sentences in `{1, 2}`.

This gives 60 policies. For each fold, the lexical verifier is trained only on
\(T_i\), and a BM25 top-10 candidate trace is written before that fold's gold
annotations are loaded. Each policy is then applied to the pooled, frozen
out-of-fold traces.

Select the policy that minimizes pooled unsupported-assertion rate, subject to
coverage of at least 0.20. Exact ties are resolved by higher coverage, higher
faithfulness, higher strict citation F1, higher thresholds, then fewer cited
sentences. This rule prevents the all-abstention policy from appearing best.

## Final evaluation

Train one verifier on the complete ordinary training split. On
`claims_dev.jsonl`, freeze a complete zero-threshold candidate trace, then
apply both the selected policy and the fixed Phase 05 policy to that same trace.
Load development gold annotations only after both decisions have frozen.

## Reproduction

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent build-index
& .\.venv\Scripts\python.exe -m evidence_agent calibrate-citation-audit
& .\.venv\Scripts\python.exe -m evidence_agent train-verifier
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-citation-audit
```

The committed outputs are `results/citation_audit_cross_validation.json` and
`results/citation_audit_dev.json`. Fold models and complete candidate traces are
local ignored artifacts under `artifacts/`; each committed report records their
paths and SHA-256 checksums.
