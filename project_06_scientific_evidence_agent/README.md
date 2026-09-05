# Scientific Evidence Verification and Citation-Audit Agent

## Objective

Evaluate whether a claim-level retrieve-rerank-verify-audit pipeline reduces
unsupported scientific assertions relative to a direct retrieval-augmented
generation baseline. The project uses SciFact for retrieval, evidence-rationale,
and stance-verification evaluation.

## Current phase

Phase 04 is complete. Its fixed corpus-only latent-semantic hybrid was a
transparent negative control: on the frozen development comparison it did not
outperform BM25. BM25 is therefore retained as the candidate retriever for the
next evidence-selection and stance-verification phase. The hybrid report and
its audit trail remain in the repository so that this decision is reproducible,
rather than silently discarded. Raw data and fitted index artifacts remain
outside version control.

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
records why BM25 remains the selected retrieval path. General `evaluate`
remains unavailable until later phases.
