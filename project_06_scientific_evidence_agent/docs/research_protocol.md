# Research protocol

## Research question

Under a fixed SciFact corpus, split, and retrieval budget, does a
retrieve-rerank-verify-audit agent reduce unsupported scientific assertions in
comparison with a direct RAG baseline?

## Hypotheses

1. Hybrid retrieval and reranking improve evidence-document Recall@k over a
   BM25-only baseline.
2. Sentence-level evidence selection plus stance verification improves strict
   evidence F1 and claim-abstract macro-F1.
3. Citation auditing reduces strict unsupported-assertion rate while retaining
   useful answer coverage.

## Experimental arms

| Arm | Workflow |
| --- | --- |
| Direct RAG | BM25 top-k abstracts -> direct verdict and citations |
| Verification agent | hybrid retrieval -> reranking -> sentence selection -> NLI -> citation audit -> verdict or abstention |

Both arms will use the same SciFact corpus, held-out claims, maximum retrieval
budget, and final response format. Random seeds, model identifiers, prompts,
hardware, package versions, and latency measurements will be retained in the
experiment manifest.

## Data-split policy

Use the provided five-fold splits for model and threshold selection. Freeze the
configuration before one final development-set evaluation. The hidden test set
may be used only for final prediction generation or a leaderboard submission,
never for local metric selection.

## Leakage policy

`evidence` and `cited_doc_ids` are evaluator-only. Runtime objects contain
only `id` and `claim`; the corpus contains only public document content. Tests
must fail if gold annotation fields enter a runtime payload.

## Statistical reporting

Report point estimates and paired bootstrap 95% confidence intervals over
claims. Report unsupported-assertion rate jointly with assertion coverage and
abstention rate so abstaining on every claim cannot appear successful.
