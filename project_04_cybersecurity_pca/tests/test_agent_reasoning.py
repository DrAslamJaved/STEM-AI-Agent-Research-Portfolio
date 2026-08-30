from dataclasses import fields, is_dataclass, replace
import hashlib
from inspect import signature
import json
from pathlib import Path

import pytest

from cyber_pca.agent_reasoning import (
    AGENT_REASONING_CRITERIA,
    AGENT_REASONING_HARD_FAILURES,
    AgentReasoningEvaluation,
    AgentReasoningSubmission,
    ReasoningCriterionAnnotation,
    evaluate_agent_reasoning,
    load_agent_reasoning_submission,
)


EXPECTED_CRITERIA = (
    "subspace_geometry",
    "anomaly_not_attack_proof",
    "normal_fit_assumption",
    "leakage_and_contamination",
    "scaling_and_threshold",
    "error_consequences",
    "empirical_validation",
)

EXPECTED_HARD_FAILURES = (
    "anomaly_claimed_as_attack_proof",
    "test_leakage_accepted",
    "frozen_metrics_misreported",
    "post_evaluation_tuning_claimed",
    "operational_deployment_recommended",
)


def test_agent_reasoning_public_interface() -> None:
    assert AGENT_REASONING_CRITERIA == EXPECTED_CRITERIA
    assert (
        AGENT_REASONING_HARD_FAILURES
        == EXPECTED_HARD_FAILURES
    )

    assert is_dataclass(
        ReasoningCriterionAnnotation
    )
    assert is_dataclass(
        AgentReasoningSubmission
    )
    assert is_dataclass(
        AgentReasoningEvaluation
    )

    assert tuple(
        field.name
        for field in fields(
            ReasoningCriterionAnnotation
        )
    ) == (
        "criterion",
        "score",
        "rationale",
        "evidence_quotes",
        "evidence_paths",
    )

    assert tuple(
        field.name
        for field in fields(
            AgentReasoningSubmission
        )
    ) == (
        "response_path",
        "response_sha256",
        "reviewer",
        "human_reviewed",
        "operational_recommendation",
        "annotations",
        "hard_failure_flags",
    )

    assert tuple(
        field.name
        for field in fields(
            AgentReasoningEvaluation
        )
    ) == (
        "response_sha256",
        "reviewer",
        "criteria",
        "total_score",
        "maximum_score",
        "score_fraction",
        "minimum_total_score",
        "zero_score_criteria",
        "hard_failures",
        "operational_recommendation",
        "passed",
    )

    assert callable(
        load_agent_reasoning_submission
    )
    assert callable(evaluate_agent_reasoning)

    load_signature = signature(
        load_agent_reasoning_submission
    )

    assert tuple(
        load_signature.parameters
    ) == ("annotation_path",)

    evaluation_signature = signature(
        evaluate_agent_reasoning
    )

    assert tuple(
        evaluation_signature.parameters
    ) == (
        "response_text",
        "submission",
        "minimum_total_score",
        "maximum_total_score",
        "require_no_zero_scores",
    )

    assert (
        evaluation_signature.parameters[
            "minimum_total_score"
        ].default
        == 12
    )
    assert (
        evaluation_signature.parameters[
            "maximum_total_score"
        ].default
        == 14
    )
    assert (
        evaluation_signature.parameters[
            "require_no_zero_scores"
        ].default
        is True
    )


def test_loads_agent_reasoning_submission(
    tmp_path: Path,
) -> None:
    response_text = (
        "PCA learns a normal linear subspace. "
        "A large residual is anomaly evidence, "
        "not proof of malicious activity."
    )

    response_sha256 = hashlib.sha256(
        response_text.encode("utf-8")
    ).hexdigest()

    annotation_path = (
        tmp_path / "annotations.json"
    )

    payload = {
        "response_path": (
            "results/"
            "phase_09_agent_reasoning_response.md"
        ),
        "response_sha256": response_sha256,
        "reviewer": "Independent human reviewer",
        "human_reviewed": True,
        "operational_recommendation": (
            "not_recommended"
        ),
        "annotations": [
            {
                "criterion": criterion,
                "score": 2,
                "rationale": (
                    f"Complete evidence for {criterion}."
                ),
                "evidence_quotes": [
                    (
                        "PCA learns a normal "
                        "linear subspace."
                    )
                ],
                "evidence_paths": [
                    (
                        "results/"
                        "unsw_nb15_evaluation.json"
                    )
                ],
            }
            for criterion in EXPECTED_CRITERIA
        ],
        "hard_failure_flags": [],
    }

    annotation_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    submission = (
        load_agent_reasoning_submission(
            annotation_path
        )
    )

    assert isinstance(
        submission,
        AgentReasoningSubmission,
    )
    assert submission.response_path == (
        "results/"
        "phase_09_agent_reasoning_response.md"
    )
    assert (
        submission.response_sha256
        == response_sha256
    )
    assert (
        submission.reviewer
        == "Independent human reviewer"
    )
    assert submission.human_reviewed is True
    assert (
        submission.operational_recommendation
        == "not_recommended"
    )
    assert len(submission.annotations) == 7
    assert all(
        isinstance(
            annotation,
            ReasoningCriterionAnnotation,
        )
        for annotation in submission.annotations
    )
    assert submission.hard_failure_flags == ()


def test_evaluates_complete_reasoning_submission() -> None:
    evidence_quotes = tuple(
        (
            f"Complete evidence for "
            f"{criterion}."
        )
        for criterion in EXPECTED_CRITERIA
    )

    response_text = "\n".join(
        evidence_quotes
    )

    response_sha256 = hashlib.sha256(
        response_text.encode("utf-8")
    ).hexdigest()

    annotations = tuple(
        ReasoningCriterionAnnotation(
            criterion=criterion,
            score=2,
            rationale=(
                f"The response completely addresses "
                f"{criterion}."
            ),
            evidence_quotes=(evidence_quote,),
            evidence_paths=(
                "results/"
                "unsw_nb15_evaluation.json",
            ),
        )
        for criterion, evidence_quote in zip(
            EXPECTED_CRITERIA,
            evidence_quotes,
            strict=True,
        )
    )

    submission = AgentReasoningSubmission(
        response_path=(
            "results/"
            "phase_09_agent_reasoning_response.md"
        ),
        response_sha256=response_sha256,
        reviewer="Independent human reviewer",
        human_reviewed=True,
        operational_recommendation=(
            "not_recommended"
        ),
        annotations=annotations,
        hard_failure_flags=(),
    )

    result = evaluate_agent_reasoning(
        response_text,
        submission,
    )

    assert isinstance(
        result,
        AgentReasoningEvaluation,
    )
    assert (
        result.response_sha256
        == response_sha256
    )
    assert (
        result.reviewer
        == "Independent human reviewer"
    )
    assert result.criteria == annotations
    assert result.total_score == 14
    assert result.maximum_score == 14
    assert result.score_fraction == 1.0
    assert result.minimum_total_score == 12
    assert result.zero_score_criteria == ()
    assert result.hard_failures == ()
    assert (
        result.operational_recommendation
        == "not_recommended"
    )
    assert result.passed is True


def test_rejects_unapproved_evidence_path() -> None:
    evidence_quotes = tuple(
        f"Evidence for {criterion}."
        for criterion in EXPECTED_CRITERIA
    )

    response_text = "\n".join(
        evidence_quotes
    )

    annotations = tuple(
        ReasoningCriterionAnnotation(
            criterion=criterion,
            score=2,
            rationale=(
                f"Complete reasoning for {criterion}."
            ),
            evidence_quotes=(quote,),
            evidence_paths=(
                "results/"
                "unsw_nb15_evaluation.json",
            ),
        )
        for criterion, quote in zip(
            EXPECTED_CRITERIA,
            evidence_quotes,
            strict=True,
        )
    )

    annotations = (
        replace(
            annotations[0],
            evidence_paths=("README.md",),
        ),
        *annotations[1:],
    )

    submission = AgentReasoningSubmission(
        response_path=(
            "results/"
            "phase_09_agent_reasoning_response.md"
        ),
        response_sha256=hashlib.sha256(
            response_text.encode("utf-8")
        ).hexdigest(),
        reviewer="Independent human reviewer",
        human_reviewed=True,
        operational_recommendation=(
            "not_recommended"
        ),
        annotations=annotations,
        hard_failure_flags=(),
    )

    with pytest.raises(
        ValueError,
        match="approved Phase 8 evidence path",
    ):
        evaluate_agent_reasoning(
            response_text,
            submission,
        )
