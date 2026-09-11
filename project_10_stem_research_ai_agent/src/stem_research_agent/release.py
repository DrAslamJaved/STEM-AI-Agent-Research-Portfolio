"""Final evaluation and release gates for the STEM Research AI Agent MVP."""
from dataclasses import asdict, dataclass
import re


CLAIM_ID_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*)\]")
LIMITATIONS = (
    "This MVP supports researcher-controlled drafting; it does not establish autonomous authorship.",
    "Evidence links and audit checks reduce risk but do not guarantee factual completeness or publication readiness.",
    "Similarity and overlap checks are safeguards, not a claim of absolute plagiarism freedom.",
)


@dataclass(frozen=True)
class DraftEvaluation:
    cited_claim_ids: tuple[str, ...]
    supported_claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]
    supported_claim_rate: float
    citation_validity_rate: float


@dataclass(frozen=True)
class ReleaseReport:
    baseline: DraftEvaluation
    controlled: DraftEvaluation
    supported_claim_rate_delta: float
    audit_passed: bool
    audit_finding_count: int
    human_review_exportable: bool
    reproducibility_passed: bool
    human_correction_count: int
    release_ready: bool
    limitations: tuple[str, ...] = LIMITATIONS

    def to_dict(self):
        return asdict(self)


def _finalizable_claim_ids(ledger) -> set[str]:
    value = getattr(ledger, "finalizable_claim_ids")
    return set(value() if callable(value) else value)


def evaluate_draft(text: str, ledger) -> DraftEvaluation:
    """Score explicit claim references against the approved evidence ledger."""
    cited = tuple(dict.fromkeys(CLAIM_ID_PATTERN.findall(text)))
    approved = _finalizable_claim_ids(ledger)
    supported = tuple(claim_id for claim_id in cited if claim_id in approved)
    unsupported = tuple(claim_id for claim_id in cited if claim_id not in approved)
    rate = len(supported) / len(cited) if cited else 0.0
    return DraftEvaluation(cited, supported, unsupported, rate, rate)


def evaluate_release(
    baseline_text: str,
    controlled_text: str,
    ledger,
    audit_report,
    review_record,
    *,
    tests_passed: bool,
    compileall_passed: bool,
    diff_check_passed: bool,
    human_correction_count: int,
) -> ReleaseReport:
    """Combine evidence, audit, human review, and reproducibility gates."""
    if human_correction_count < 0:
        raise ValueError("human_correction_count cannot be negative.")

    baseline = evaluate_draft(baseline_text, ledger)
    controlled = evaluate_draft(controlled_text, ledger)
    audit_passed = bool(getattr(audit_report, "passed"))
    findings = getattr(audit_report, "findings", ())
    review_exportable = bool(review_record.exportable())
    reproducibility = tests_passed and compileall_passed and diff_check_passed
    release_ready = (
        controlled.citation_validity_rate == 1.0
        and audit_passed
        and review_exportable
        and reproducibility
    )
    return ReleaseReport(
        baseline=baseline,
        controlled=controlled,
        supported_claim_rate_delta=controlled.supported_claim_rate - baseline.supported_claim_rate,
        audit_passed=audit_passed,
        audit_finding_count=len(findings),
        human_review_exportable=review_exportable,
        reproducibility_passed=reproducibility,
        human_correction_count=human_correction_count,
        release_ready=release_ready,
    )
