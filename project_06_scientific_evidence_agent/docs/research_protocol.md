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

## Phase 04 fixed hybrid design

The first hybrid implementation is deliberately corpus-only: TF-IDF followed by
128-dimensional truncated SVD, merged with the frozen BM25 rank list using
reciprocal-rank fusion (`k=60`). A transparent candidate reranker uses fixed
weights recorded in the result report. These choices are not fitted to SciFact
gold annotations. This phase tests retrieval improvement only; it does not yet
make a claim about final stance correctness or unsupported assertions.

## Phase 04 outcome and downstream selection

The pre-specified Phase 04 hybrid comparison was retained even though it was
unfavourable: its development-set Recall@k and MRR were lower than the committed
BM25 baseline. This is a result, not a failed evaluation. The project therefore
uses BM25 as the candidate generator in the next phase, and keeps the hybrid
report as an auditable negative control. The report is not replaced by a
post-evaluation tuned hybrid result. A later learned dense or cross-encoder
retriever, if introduced, must be selected with the provided five-fold splits
before one new frozen development evaluation.

## Phase 05 lexical verification baseline

The first verifier is a deliberately transparent baseline rather than an
opaque pretrained NLI model. It fits a three-way claim/document classifier and
a binary claim/sentence selector solely from `claims_train.jsonl`. Pair inputs
are represented by separate claim and evidence TF-IDF vectors, their
element-wise overlap, and their absolute difference. Each model uses fixed
logistic-regression settings, `max_features=40000`, and `random_seed=20260904`.

The runtime uses only BM25 top-10 corpus documents, the trained local artifact,
claim text, and corpus sentences. It emits an assertive decision only if the
geometric mean of the best assertion probability and sentence-evidence
probability is at least 0.45; the sentence threshold is 0.50 and at most two
sentences are cited. Development cited-document labels, evidence rationales,
and claim verdicts are read only after all runtime traces freeze to an ignored
local artifact. The version-controlled result retains only one decision per
claim plus the full trace's SHA-256 and metadata.

The Phase 05 report distinguishes: (1) controlled three-way stance macro-F1 on
cited-document pairs; (2) end-to-end claim macro-F1 after BM25 retrieval;
(3) sentence evidence F1; (4) strict citation correctness; and (5) faithfulness,
coverage, unsupported-assertion rate, and latency. It is not a superiority
claim against direct RAG. The next phase will select an abstention/citation
audit policy using the provided five-fold splits before a new frozen
development comparison.

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
