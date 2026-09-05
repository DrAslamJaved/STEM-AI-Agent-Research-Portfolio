# Phase 05 evidence selection and stance-verification protocol

## Objective

Establish a reproducible, leakage-safe baseline for the two parts of evidence
verification that retrieval alone cannot perform:

1. distinguish `SUPPORT`, `CONTRADICT`, and `NO_EVIDENCE` for a claim/document
   pair; and
2. identify the sentence(s) that justify an assertive claim decision.

## Training boundary

`train-verifier` consumes only the public corpus and
`claims_train.jsonl`. Evidence labels and rationale sentences are legitimate
supervision within that train split. The fitted `joblib` bundle is a local,
ignored artifact and must be loaded only from a trusted local path.

No development claim, development evidence label, or hidden-test data may be
used to fit the vectorizers, classifiers, or thresholds.

## Fixed baseline

Both models use word unigram/bigram TF-IDF vectors. A pair is represented by
the claim vector, evidence vector, element-wise overlap, and absolute
difference. Logistic regression uses balanced classes, `max_iter=1000`, and
`random_seed=20260904`.

The runtime starts with BM25 top-10 documents. It scores all public abstract
sentences in those documents, retains up to two sentences with probability at
least 0.50, and emits an assertive verdict only if the geometric mean of the
best assertion and evidence probabilities is at least 0.45. Otherwise it emits
`NO_EVIDENCE` with no citation.

## Evaluation boundary

The complete runtime trace freezes to an ignored local artifact first. Only
then does the evaluator load:

- cited-document pairs for controlled stance macro-F1;
- complete SciFact rationale sets for strict citation correctness and
  sentence-level evidence F1; and
- claim verdicts for claim macro-F1, faithfulness, coverage, and
  unsupported-assertion rate.

`unsupported_assertion_rate` counts an assertive decision as unsupported when
its claim stance is wrong or none of its selected citation sentences overlaps a
gold sentence with matching stance. This deliberately prevents an unsupported
citation from being treated as a correct answer merely because the claim label
happens to match.

## Report retention

`results/verification_dev.json` is deliberately compact and suitable for Git:
it retains the fixed settings, model and corpus provenance, metrics, and one
claim-level audit decision (verdict, confidence, and citations) per development
claim. The full candidate-level and sentence-level diagnostic trace is written
to `artifacts/verification_dev_trace.json`, which is ignored by Git. The compact
report records the local trace path, SHA-256, schema version, and trace count so
the detailed trace can be verified when it is available without committing an
unnecessarily large generated file.
