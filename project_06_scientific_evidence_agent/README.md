# Scientific Evidence Verification and Citation-Audit Agent

## Objective

Evaluate whether a claim-level retrieve-rerank-verify-audit pipeline reduces
unsupported scientific assertions relative to a direct retrieval-augmented
generation baseline. The project uses SciFact for retrieval, evidence-rationale,
and stance-verification evaluation.

## Current phase

Phase 05 adds a train-only lexical verification baseline: TF-IDF relation
features and logistic regression classify document stance, while a separate
sentence model selects citable evidence. BM25 remains the candidate retriever,
because the Phase 04 hybrid did not outperform it. The verifier reports
three-way stance macro-F1, claim-classification macro-F1, evidence F1,
citation correctness, faithfulness, coverage, unsupported-assertion rate, and
latency. The first lexical result is retained as a baseline rather than treated
as a final scientific-verification system. Raw data and fitted model artifacts
remain outside version control.

## Non-negotiable evaluation rule

The runtime agent may use only a claim's identifier and text. SciFact gold
`evidence` and `cited_doc_ids` are evaluator-only fields and must never reach
retrieval, reranking, or verification code.

## Quick start

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m evidence_agent acquire-data
& .\.venv\Scripts\python.exe -m evidence_agent validate-data
& .\.venv\Scripts\python.exe -m evidence_agent build-index
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-retrieval
& .\.venv\Scripts\python.exe -m evidence_agent build-semantic-index
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-hybrid-retrieval
& .\.venv\Scripts\python.exe -m evidence_agent train-verifier
& .\.venv\Scripts\python.exe -m evidence_agent evaluate-verifier
& .\.venv\Scripts\python.exe -m evidence_agent contract
```

## Repository layout

```text
configs/       Deterministic experiment configuration
data/          Raw, interim, and processed data boundaries
docs/          Research protocol, architecture, and evaluation contract
src/           Installable Python package
tests/         Unit and integration tests
validation/    Provenance and validation evidence
reports/       Human-readable experimental reports
results/       Machine-readable final experimental outputs
agent_trace/   Phase-level decision and verification trace
```

## Planned commands

`acquire-data` downloads the official SciFact release, records SHA-256
provenance, and safely extracts it. `validate-data` verifies the corpus, claim
splits, rationale references, and five-fold layout before any model may use the
dataset. `build-index` creates a fixed BM25 baseline from public corpus text;
`evaluate-retrieval` freezes retrieval outputs before accessing evaluator-only
gold evidence. `build-semantic-index` fits a deterministic TF-IDF + LSA index
from the public corpus, while `evaluate-hybrid-retrieval` combines BM25 and LSA
with fixed reciprocal-rank fusion and a documented candidate reranker. The
result is retained as a diagnostic comparator; `reports/phase_04_hybrid_retrieval.md`
records why BM25 remains the selected retrieval path. `train-verifier` fits
only on `claims_train.jsonl`; `evaluate-verifier` freezes BM25-driven runtime
decisions into an ignored full diagnostic trace before reading development
evidence labels for offline scoring. Its committed result file remains compact:
one auditable verdict and citation record per claim plus the full trace's local
path and SHA-256. General `evaluate` remains unavailable until later phases.
