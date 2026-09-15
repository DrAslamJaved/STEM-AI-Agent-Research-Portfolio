"""Deterministic, evidence-constrained drafting for the real Davis DTI workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_CONTEXT_CLAIMS = {"C01", "C02"}
REQUIRED_SPLITS = ("pair_random", "cold_drug", "cold_target")


def _load_json(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load constrained-draft input: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Constrained-draft inputs must be JSON objects.")
    return value


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Model report is missing numeric {label}.")
    return float(value)


def _verified_references(ledger: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    finalizable = set(ledger.get("finalizable_claim_ids", []))
    if not REQUIRED_CONTEXT_CLAIMS <= finalizable:
        missing = ", ".join(sorted(REQUIRED_CONTEXT_CLAIMS - finalizable))
        raise ValueError(f"Literature ledger lacks required finalizable claim(s): {missing}")
    sources = {source.get("source_id"): source for source in ledger.get("sources", []) if isinstance(source, Mapping)}
    links = ledger.get("links", [])
    references: dict[str, Mapping[str, Any]] = {}
    for claim_id in REQUIRED_CONTEXT_CLAIMS:
        supporting = [link for link in links if isinstance(link, Mapping)
                      and link.get("claim_id") == claim_id and link.get("entailment") == "supports"]
        if not supporting:
            raise ValueError(f"Literature ledger lacks a supporting link for {claim_id}.")
        source = sources.get(supporting[0].get("source_id"))
        if not source or source.get("verification_status") != "VERIFIED":
            raise ValueError(f"Literature ledger lacks a verified supporting source for {claim_id}.")
        references[claim_id] = source
    return references


def _result_rows(report: Mapping[str, Any]) -> list[dict[str, float | str]]:
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, Mapping):
        raise ValueError("Model report is missing outcomes.")
    rows: list[dict[str, float | str]] = []
    for split in REQUIRED_SPLITS:
        outcome = outcomes.get(split)
        if not isinstance(outcome, Mapping) or not isinstance(outcome.get("models"), Mapping):
            raise ValueError(f"Model report is missing the {split} condition.")
        models = outcome["models"]
        forest, prevalence = models.get("random_forest"), models.get("prevalence")
        if not isinstance(forest, Mapping) or not isinstance(prevalence, Mapping):
            raise ValueError(f"Model report is missing required baselines for {split}.")
        forest_pr = _number(forest.get("pr_auc"), f"{split} random-forest PR-AUC")
        prevalence_pr = _number(prevalence.get("pr_auc"), f"{split} prevalence PR-AUC")
        if not all(0 <= value <= 1 for value in (forest_pr, prevalence_pr)):
            raise ValueError(f"Model report contains an invalid PR-AUC for {split}.")
        rows.append({"split": split, "forest_pr_auc": forest_pr, "prevalence_pr_auc": prevalence_pr,
                     "improvement": forest_pr - prevalence_pr})
    return rows


@dataclass(frozen=True)
class ConstrainedDavisDraft:
    markdown: str
    trace: dict[str, Any]


def build_constrained_davis_draft(model_report: Mapping[str, Any], literature_ledger: Mapping[str, Any]) -> ConstrainedDavisDraft:
    """Create a review draft from approved context claims and recorded model metrics."""
    references = _verified_references(literature_ledger)
    rows = _result_rows(model_report)
    dataset_id = model_report.get("dataset_id")
    source_commit = model_report.get("source_commit")
    if not isinstance(dataset_id, str) or not dataset_id.strip() or not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("Model report requires dataset_id and source_commit provenance.")
    results_table = "\n".join(
        f"| {row['split'].replace('_', ' ')} | {row['forest_pr_auc']:.3f} | {row['prevalence_pr_auc']:.3f} | {row['improvement']:.3f} |"
        for row in rows
    )
    markdown = f"""# Evidence-Constrained Baseline Evaluation on the Davis DTI Benchmark

## Abstract

This researcher-controlled draft reports fixed-feature baseline evaluation on the
`{dataset_id}` dataset. The workflow preserves dataset provenance, evaluates
pair-random and cold-start conditions separately, and constrains contextual
statements to Crossref-verified, human-approved evidence. It does not claim
clinical validity, experimental binding confirmation, or state-of-the-art DTI
performance.

## Introduction

The Davis study profiled kinase-inhibitor interactions across a broad kinase
panel [@C01]. DeepDTA describes binding-affinity prediction from compound and
protein sequence representations [@C02]. These sources provide benchmark and
method context; they do not validate the present models' conclusions.

## Methods

The recorded source commit was `{source_commit}`. Fixed SMILES and protein
composition features were evaluated using a training-prevalence reference,
class-balanced logistic regression, and a class-balanced random forest. The
primary ranking metric reported here is PR-AUC because the positive class is
uncommon. Pair-random, cold-drug, and cold-target splits are reported
separately; pair-random identities overlap across partitions and therefore do
not establish cold-start generalization.

## Results

The table reports recorded PR-AUC values from the committed model report.

| Split | Random-forest PR-AUC | Prevalence PR-AUC | Difference |
| --- | ---: | ---: | ---: |
{results_table}

Across the recorded conditions, the random forest had a higher PR-AUC than the
training-prevalence reference. The cold-drug result should be interpreted as a
test of unseen-compound generalization, while the cold-target result addresses
unseen-target generalization.

## Discussion

These results establish reproducible baseline behavior for this specific
dataset, representation, split design, and seed. They do not identify causal
binding mechanisms, demonstrate prospective performance, or establish utility
for drug discovery decisions. Any comparison with other studies requires a
matched task definition, data split, and evaluation protocol.

## Evidence and review limitations

The contextual claims above are traceable to the approved claim IDs `C01` and
`C02`. Crossref verification confirms bibliographic identity only; a researcher
must still assess source entailment, quality, completeness, and manuscript
suitability before release.

## References

1. {references['C01']['title']} ({references['C01']['year']}). DOI: {references['C01']['doi']}.
2. {references['C02']['title']} ({references['C02']['year']}). DOI: {references['C02']['doi']}.
"""
    trace = {
        "dataset_id": dataset_id,
        "source_commit": source_commit,
        "context_claim_ids": sorted(REQUIRED_CONTEXT_CLAIMS),
        "result_claim_ids": [f"R{index:02d}" for index in range(1, len(rows) + 1)],
        "results": rows,
        "limitations": [
            "Crossref verification establishes bibliographic identity, not scientific entailment.",
            "Pair-random results do not establish cold-start generalization.",
            "The draft is not a clinical, prospective, or state-of-the-art performance claim.",
        ],
    }
    return ConstrainedDavisDraft(markdown, trace)


def build_draft_from_files(model_report_path: str | Path, literature_ledger_path: str | Path) -> ConstrainedDavisDraft:
    return build_constrained_davis_draft(_load_json(model_report_path), _load_json(literature_ledger_path))
