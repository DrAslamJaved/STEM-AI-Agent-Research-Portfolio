"""Deterministic input validation for Phase 2."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable, Mapping

REQUIRED_DTI_FIELDS = ("drug_id", "target_id", "label")

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    row_index: int | None = None
    severity: str = "error"

@dataclass
class ValidationReport:
    subject: str
    checked_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, int | float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, code: str, message: str, *, row_index: int | None = None, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(code, message, row_index, severity))

    def to_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "checked_count": self.checked_count, "is_valid": self.is_valid,
                "metrics": self.metrics, "issues": [issue.__dict__ for issue in self.issues]}

def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())

def _binary_label(value: Any) -> bool:
    return value in (0, 1, 0.0, 1.0, "0", "1")

def validate_dti_records(records: Iterable[Mapping[str, Any]]) -> ValidationReport:
    """Validate DTI records without changing their order or contents."""
    rows = list(records)
    report = ValidationReport(subject="dti_records", checked_count=len(rows))
    pair_counts: dict[tuple[str, str], int] = {}
    drugs: set[str] = set()
    targets: set[str] = set()
    if not rows:
        report.add("empty_dataset", "At least one DTI record is required.")
        return report
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            report.add("invalid_row", "Each DTI record must be a mapping.", row_index=index)
            continue
        missing = [field for field in REQUIRED_DTI_FIELDS if _missing(row.get(field))]
        if missing:
            report.add("missing_required_field", f"Missing required field(s): {', '.join(missing)}.", row_index=index)
            continue
        if not _binary_label(row["label"]):
            report.add("invalid_label", "DTI label must be binary: 0 or 1.", row_index=index)
        drug, target = str(row["drug_id"]).strip(), str(row["target_id"]).strip()
        drugs.add(drug); targets.add(target)
        key = (drug, target)
        pair_counts[key] = pair_counts.get(key, 0) + 1
    duplicate_pairs = sum(count - 1 for count in pair_counts.values() if count > 1)
    if duplicate_pairs:
        report.add("duplicate_drug_target_pair", f"{duplicate_pairs} duplicate drug-target pair(s) detected.", severity="warning")
    if len(drugs) < 2 or len(targets) < 2:
        report.add("split_leakage_risk", "Fewer than two unique drugs or targets; cold-start evaluation is impossible.", severity="warning")
    report.metrics.update(n_records=len(rows), n_unique_drugs=len(drugs), n_unique_targets=len(targets),
                          n_duplicate_pairs=duplicate_pairs)
    return report

def validate_fuzzy_membership_vector(values: Iterable[Any]) -> ValidationReport:
    """Validate one finite, non-empty fuzzy membership vector."""
    memberships = list(values)
    report = ValidationReport(subject="fuzzy_membership_vector", checked_count=len(memberships))
    if not memberships:
        report.add("empty_universe", "The fuzzy universe must be non-empty.")
        return report
    for index, value in enumerate(memberships):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            report.add("non_numeric_membership", "Membership must be numeric.", row_index=index)
            continue
        numeric = float(value)
        if not isfinite(numeric):
            report.add("non_finite_membership", "Membership must be finite.", row_index=index)
        elif not 0.0 <= numeric <= 1.0:
            report.add("membership_out_of_bounds", "Membership must lie in [0, 1].", row_index=index)
    report.metrics["cardinality"] = sum(float(value) for value in memberships
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(float(value)))
    return report
