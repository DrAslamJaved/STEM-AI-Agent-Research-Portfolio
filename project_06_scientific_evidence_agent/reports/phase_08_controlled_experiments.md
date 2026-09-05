# Phase 08 — Controlled direct-RAG vs audited-agent experiment

**This is a held-out development evaluation, not an independent test.**

Both arms consumed the same frozen BM25/verifier trace. The direct-RAG arm used
the first retrieved document with up to three top-scoring rationale sentences and
no audit thresholds; the audited arm applied the frozen Phase 06 policy.

Result JSON SHA-256: `32fe3b1d7c6886db580b08ad9ffb694f85961bd7b19cafb980e1507a2c081b5d`
Raw trace SHA-256: `43679c7ec8b69d4b21666434039e477901a9cbe64964772fafb146919bdccdc9`

## Official SciFact-compatible scoring

| Metric | Direct RAG | Audited agent | Delta | 95% paired-bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Abstract-level F1 | 0.1925 | 0.0565 | -0.1360 | [-0.1871, -0.0870] |
| Sentence-level F1 | 0.1043 | 0.0428 | -0.0614 | [-0.0946, -0.0288] |

## Adversarial scoring checks

The official evaluator regression suite rejects a wrong document, wrong stance,
and incomplete multi-sentence rationale.

All adversarial scoring checks passed: `True`.

Quality claims must be made only when the corresponding paired confidence interval
excludes zero. The development result must not be represented as an independent test.
