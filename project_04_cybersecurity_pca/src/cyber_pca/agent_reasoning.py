"""Structured evaluation of AI-agent reasoning quality."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


AGENT_REASONING_CRITERIA = (
    "subspace_geometry",
    "anomaly_not_attack_proof",
    "normal_fit_assumption",
    "leakage_and_contamination",
    "scaling_and_threshold",
    "error_consequences",
    "empirical_validation",
)

AGENT_REASONING_HARD_FAILURES = (
    "anomaly_claimed_as_attack_proof",
    "test_leakage_accepted",
    "frozen_metrics_misreported",
    "post_evaluation_tuning_claimed",
    "operational_deployment_recommended",
)

AGENT_REASONING_EVIDENCE_PATHS = (
    "results/unsw_nb15_evaluation.json",
    "reports/tables/unsw_nb15_metrics.csv",
    (
        "reports/tables/"
        "unsw_nb15_attack_category_metrics.csv"
    ),
    "agent_trace/phase_08.md",
    "docs/critical_reasoning.md",
)


@dataclass(frozen=True, slots=True)
class ReasoningCriterionAnnotation:
    """Human-reviewed score and evidence for one criterion."""

    criterion: str
    score: int
    rationale: str
    evidence_quotes: tuple[str, ...]
    evidence_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentReasoningSubmission:
    """Structured human-reviewed reasoning submission."""

    response_path: str
    response_sha256: str
    reviewer: str
    human_reviewed: bool
    operational_recommendation: str
    annotations: tuple[
        ReasoningCriterionAnnotation,
        ...,
    ]
    hard_failure_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentReasoningEvaluation:
    """Deterministic result of applying the reasoning rubric."""

    response_sha256: str
    reviewer: str
    criteria: tuple[
        ReasoningCriterionAnnotation,
        ...,
    ]
    total_score: int
    maximum_score: int
    score_fraction: float
    minimum_total_score: int
    zero_score_criteria: tuple[str, ...]
    hard_failures: tuple[str, ...]
    operational_recommendation: str
    passed: bool


def _require_text(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} must not be empty."
        )

    return normalized


def _require_string_tuple(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{field_name} must be a sequence."
        )

    return tuple(
        _require_text(
            item,
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )


def _parse_annotation(
    value: Any,
    index: int,
) -> ReasoningCriterionAnnotation:
    if not isinstance(value, dict):
        raise ValueError(
            f"annotations[{index}] must be an object."
        )

    criterion = _require_text(
        value.get("criterion"),
        f"annotations[{index}].criterion",
    )

    score = value.get("score")

    if (
        not isinstance(score, int)
        or isinstance(score, bool)
        or score not in (0, 1, 2)
    ):
        raise ValueError(
            f"annotations[{index}].score must be "
            "0, 1, or 2."
        )

    return ReasoningCriterionAnnotation(
        criterion=criterion,
        score=score,
        rationale=_require_text(
            value.get("rationale"),
            f"annotations[{index}].rationale",
        ),
        evidence_quotes=_require_string_tuple(
            value.get("evidence_quotes"),
            (
                f"annotations[{index}]"
                ".evidence_quotes"
            ),
        ),
        evidence_paths=_require_string_tuple(
            value.get("evidence_paths"),
            (
                f"annotations[{index}]"
                ".evidence_paths"
            ),
        ),
    )


def load_agent_reasoning_submission(
    annotation_path: str | Path,
) -> AgentReasoningSubmission:
    """Load a structured human-reviewed annotation record."""

    path = Path(annotation_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Annotation file does not exist: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exception:
        raise ValueError(
            "Annotation file must contain valid JSON."
        ) from exception

    if not isinstance(payload, dict):
        raise ValueError(
            "Annotation JSON must be an object."
        )

    annotation_values = payload.get(
        "annotations"
    )

    if not isinstance(annotation_values, list):
        raise ValueError(
            "annotations must be a list."
        )

    human_reviewed = payload.get(
        "human_reviewed"
    )

    if not isinstance(human_reviewed, bool):
        raise ValueError(
            "human_reviewed must be Boolean."
        )

    return AgentReasoningSubmission(
        response_path=_require_text(
            payload.get("response_path"),
            "response_path",
        ),
        response_sha256=_require_text(
            payload.get("response_sha256"),
            "response_sha256",
        ),
        reviewer=_require_text(
            payload.get("reviewer"),
            "reviewer",
        ),
        human_reviewed=human_reviewed,
        operational_recommendation=(
            _require_text(
                payload.get(
                    "operational_recommendation"
                ),
                "operational_recommendation",
            )
        ),
        annotations=tuple(
            _parse_annotation(
                annotation,
                index,
            )
            for index, annotation in enumerate(
                annotation_values
            )
        ),
        hard_failure_flags=(
            _require_string_tuple(
                payload.get(
                    "hard_failure_flags"
                ),
                "hard_failure_flags",
            )
        ),
    )


def evaluate_agent_reasoning(
    response_text: str,
    submission: AgentReasoningSubmission,
    *,
    minimum_total_score: int = 12,
    maximum_total_score: int = 14,
    require_no_zero_scores: bool = True,
) -> AgentReasoningEvaluation:
    """Evaluate a structured reasoning submission."""

    if (
        not isinstance(response_text, str)
        or not response_text.strip()
    ):
        raise ValueError(
            "response_text must be a nonempty string."
        )

    if not isinstance(
        submission,
        AgentReasoningSubmission,
    ):
        raise TypeError(
            "submission must be an "
            "AgentReasoningSubmission."
        )

    if (
        not isinstance(maximum_total_score, int)
        or isinstance(maximum_total_score, bool)
        or maximum_total_score
        != len(AGENT_REASONING_CRITERIA) * 2
    ):
        raise ValueError(
            "maximum_total_score must equal 14."
        )

    if (
        not isinstance(minimum_total_score, int)
        or isinstance(minimum_total_score, bool)
        or not (
            0
            <= minimum_total_score
            <= maximum_total_score
        )
    ):
        raise ValueError(
            "minimum_total_score must be between "
            "0 and maximum_total_score."
        )

    if not isinstance(
        require_no_zero_scores,
        bool,
    ):
        raise ValueError(
            "require_no_zero_scores must be Boolean."
        )

    expected_hash = hashlib.sha256(
        response_text.encode("utf-8")
    ).hexdigest()

    if submission.response_sha256 != expected_hash:
        raise ValueError(
            "response_sha256 does not match "
            "response_text."
        )

    _require_text(
        submission.response_path,
        "submission.response_path",
    )
    _require_text(
        submission.reviewer,
        "submission.reviewer",
    )
    _require_text(
        submission.operational_recommendation,
        "submission.operational_recommendation",
    )

    if submission.human_reviewed is not True:
        raise ValueError(
            "The submission must be human reviewed."
        )

    for index, annotation in enumerate(
        submission.annotations
    ):
        if not isinstance(
            annotation,
            ReasoningCriterionAnnotation,
        ):
            raise TypeError(
                f"annotations[{index}] has an "
                "invalid type."
            )

    observed_criteria = tuple(
        annotation.criterion
        for annotation in submission.annotations
    )

    if (
        observed_criteria
        != AGENT_REASONING_CRITERIA
    ):
        raise ValueError(
            "Annotations must contain every criterion "
            "once and in canonical order."
        )

    for index, annotation in enumerate(
        submission.annotations
    ):
        if (
            not isinstance(annotation.score, int)
            or isinstance(annotation.score, bool)
            or annotation.score not in (0, 1, 2)
        ):
            raise ValueError(
                f"annotations[{index}].score must be "
                "0, 1, or 2."
            )

        _require_text(
            annotation.rationale,
            f"annotations[{index}].rationale",
        )

        if annotation.score > 0:
            if not annotation.evidence_quotes:
                raise ValueError(
                    f"annotations[{index}] requires "
                    "at least one evidence quote."
                )

            if not annotation.evidence_paths:
                raise ValueError(
                    f"annotations[{index}] requires "
                    "at least one evidence path."
                )

        for quote_index, quote in enumerate(
            annotation.evidence_quotes
        ):
            normalized_quote = _require_text(
                quote,
                (
                    f"annotations[{index}]"
                    f".evidence_quotes[{quote_index}]"
                ),
            )

            if normalized_quote not in response_text:
                raise ValueError(
                    f"annotations[{index}] contains "
                    "an evidence quote absent from "
                    "the response."
                )

        for path_index, evidence_path in enumerate(
            annotation.evidence_paths
        ):
            normalized_path = _require_text(
                evidence_path,
                (
                    f"annotations[{index}]"
                    f".evidence_paths[{path_index}]"
                ),
            )

            normalized_evidence_path = (
                normalized_path.replace(
                    "\\",
                    "/",
                )
            )

            if (
                normalized_evidence_path
                not in AGENT_REASONING_EVIDENCE_PATHS
            ):
                raise ValueError(
                    f"Evidence path is not an "
                    f"approved Phase 8 evidence path: "
                    f"{normalized_evidence_path}"
                )

            path = Path(
                normalized_evidence_path
            )

            if (
                not path.is_file()
                or path.stat().st_size == 0
            ):
                raise ValueError(
                    f"Evidence path is missing or "
                    f"empty: {path}"
                )

    if len(
        set(submission.hard_failure_flags)
    ) != len(submission.hard_failure_flags):
        raise ValueError(
            "hard_failure_flags must be unique."
        )

    unknown_hard_failures = tuple(
        failure
        for failure in submission.hard_failure_flags
        if failure
        not in AGENT_REASONING_HARD_FAILURES
    )

    if unknown_hard_failures:
        raise ValueError(
            "Unknown hard failure flags: "
            f"{unknown_hard_failures}"
        )

    hard_failure_set = set(
        submission.hard_failure_flags
    )

    if (
        submission.operational_recommendation
        != "not_recommended"
    ):
        hard_failure_set.add(
            "operational_deployment_recommended"
        )

    hard_failures = tuple(
        failure
        for failure in AGENT_REASONING_HARD_FAILURES
        if failure in hard_failure_set
    )

    total_score = sum(
        annotation.score
        for annotation in submission.annotations
    )

    zero_score_criteria = tuple(
        annotation.criterion
        for annotation in submission.annotations
        if annotation.score == 0
    )

    score_requirement_passed = (
        total_score >= minimum_total_score
    )

    zero_score_requirement_passed = (
        not require_no_zero_scores
        or not zero_score_criteria
    )

    passed = (
        score_requirement_passed
        and zero_score_requirement_passed
        and not hard_failures
        and (
            submission.operational_recommendation
            == "not_recommended"
        )
    )

    return AgentReasoningEvaluation(
        response_sha256=expected_hash,
        reviewer=submission.reviewer,
        criteria=submission.annotations,
        total_score=total_score,
        maximum_score=maximum_total_score,
        score_fraction=(
            total_score / maximum_total_score
        ),
        minimum_total_score=minimum_total_score,
        zero_score_criteria=zero_score_criteria,
        hard_failures=hard_failures,
        operational_recommendation=(
            submission.operational_recommendation
        ),
        passed=passed,
    )


__all__ = [
    "AGENT_REASONING_CRITERIA",
    "AGENT_REASONING_EVIDENCE_PATHS",
    "AGENT_REASONING_HARD_FAILURES",
    "AgentReasoningEvaluation",
    "AgentReasoningSubmission",
    "ReasoningCriterionAnnotation",
    "evaluate_agent_reasoning",
    "load_agent_reasoning_submission",
]
