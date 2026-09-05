# Phase 01 — Contract-first foundation

## Decision

Start with an installable Python package and explicit runtime/evaluator boundary
before acquiring data or selecting models.

## Rationale

SciFact exposes gold annotations alongside claim text. Without an explicit
boundary, accidentally passing `evidence` or `cited_doc_ids` into retrieval or
verification can create invalidly optimistic results.

## Evidence expected before commit

1. Editable installation in a dedicated Python 3.12 virtual environment.
2. Passing unit tests for the CLI, domain objects, and leakage contract.
3. Passing source/test compilation.
4. Clean `git diff --check` and path-scoped staging only for Project 06.
