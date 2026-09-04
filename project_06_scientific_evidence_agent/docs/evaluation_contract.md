# Evaluation contract

## Native SciFact measures

| Metric | Definition |
| --- | --- |
| Evidence-document Recall@k | Fraction of gold evidence documents retrieved in the top k documents for a claim. |
| Strict abstract F1 | A predicted citation is correct only when document, stance, and a complete gold rationale set are correct. |
| Strict sentence F1 | Sentence-level precision and recall under the official complete-rationale rule. |
| Claim-abstract macro-F1 | Macro-F1 for `SUPPORT`, `CONTRADICT`, and `NO_EVIDENCE` on labelled candidate claim-abstract pairs. |

## Audit measures

| Metric | Definition |
| --- | --- |
| Citation correctness | Strict abstract-level precision and F1. |
| Gold-grounded faithfulness | Proportion of accepted citations containing a complete gold rationale with matching stance. |
| Unsupported-assertion rate | Incorrect non-abstaining verdicts divided by all non-abstaining verdicts. |
| Coverage | Proportion of claims receiving an assertive verdict rather than abstention. |
| Cost and latency | Per-claim wall-clock time, retrieved documents, candidate sentences, model calls, and hardware details. |

## Acceptance conditions

The final report will not make a superiority claim from a single favourable
metric. It must show the direct-RAG and audited-agent values, confidence
intervals, coverage, abstention, cost, and a representative error analysis.
