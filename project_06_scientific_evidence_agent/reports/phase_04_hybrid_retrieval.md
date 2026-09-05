# Phase 04 — Hybrid retrieval result

## Scope

This report compares the committed BM25 development baseline with one fixed,
corpus-only hybrid configuration on the same 300 SciFact development claims.
The runtime ranked documents using claim identifiers and text only. Gold
evidence was loaded only after the rankings were frozen.

The hybrid configuration used TF-IDF with unigrams and bigrams, 128-dimensional
truncated SVD (`seed=20260904`), 50 candidates from each retriever,
reciprocal-rank fusion (`k=60`), and a transparent reranker. Its fixed weights
were RRF 0.45, semantic score 0.35, BM25 score 0.15, and title-term coverage
0.05. It was not trained on SciFact claims, evidence, cited documents, or
labels.

## Results

| Metric | BM25 | Hybrid | Hybrid − BM25 |
| --- | ---: | ---: | ---: |
| Claim Recall@1 | 0.7074 | 0.4415 | -0.2660 |
| Claim Recall@3 | 0.8404 | 0.6383 | -0.2021 |
| Claim Recall@5 | 0.8883 | 0.7181 | -0.1702 |
| Claim Recall@10 | 0.9415 | 0.7926 | -0.1489 |
| Evidence-document Recall@1 | 0.6364 | 0.3971 | -0.2392 |
| Evidence-document Recall@3 | 0.7751 | 0.5789 | -0.1962 |
| Evidence-document Recall@5 | 0.8325 | 0.6651 | -0.1675 |
| Evidence-document Recall@10 | 0.8995 | 0.7560 | -0.1435 |
| Mean reciprocal rank | 0.7852 | 0.5557 | -0.2295 |

There were 188 claims with gold evidence documents and 112 claims without
gold evidence. The corpus SHA-256 was
`b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62`.
The hybrid report records the complete frozen rankings, index metadata, and
the SHA-256 of the BM25 report used for comparison.

## Interpretation and decision

The evidence does not support the Phase 04 hybrid as an improvement over BM25.
That is an informative result: adding an unsupervised semantic component and a
heuristic reranker can introduce ranking errors even when the implementation is
leakage-safe and fully reproducible. The project retains this comparator rather
than discarding it, and uses BM25 as the candidate retriever for the next phase.

This does not rule out all dense or learned rerankers. Any later alternative
must be selected with the supplied five-fold SciFact splits and then evaluated
once on a frozen development split. It must also improve the later
evidence/stance/audit measures, not only a retrieval metric.

## Reproduction

From the project directory after acquiring and validating SciFact:

```powershell
& .\.venv\Scripts\python.exe -m evidence_agent build-index
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-retrieval
& .\.venv\Scripts\python.exe -m evidence_agent build-semantic-index
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-hybrid-retrieval
```

The final command writes `results/hybrid_retrieval_dev.json`. It is expected
to reproduce the direction of the reported comparison with the recorded
package versions and corpus fingerprint.
