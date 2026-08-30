from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from cyber_pca.agent_reasoning import (
    AGENT_REASONING_CRITERIA,
    AGENT_REASONING_HARD_FAILURES,
    AgentReasoningSubmission,
    ReasoningCriterionAnnotation,
    evaluate_agent_reasoning,
    load_agent_reasoning_submission,
)


def _valid_submission() -> tuple[
    str,
    AgentReasoningSubmission,
]:
    quotes = tuple(
        f"Evidence for {criterion}."
        for criterion in AGENT_REASONING_CRITERIA
    )

    response_text = "\n".join(quotes)

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
            AGENT_REASONING_CRITERIA,
            quotes,
            strict=True,
        )
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

    return response_text, submission


def test_rejects_invalid_annotation_runtime_type() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        annotations=(
            object(),
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        TypeError,
        match="invalid type",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_rejects_empty_operational_recommendation() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        operational_recommendation="",
    )

    with pytest.raises(
        ValueError,
        match="operational_recommendation",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def _valid_annotation_payload() -> dict[str, object]:
    response_text, submission = (
        _valid_submission()
    )

    return {
        "response_path": submission.response_path,
        "response_sha256": hashlib.sha256(
            response_text.encode("utf-8")
        ).hexdigest(),
        "reviewer": submission.reviewer,
        "human_reviewed": True,
        "operational_recommendation": (
            submission.operational_recommendation
        ),
        "annotations": [
            {
                "criterion": annotation.criterion,
                "score": annotation.score,
                "rationale": annotation.rationale,
                "evidence_quotes": list(
                    annotation.evidence_quotes
                ),
                "evidence_paths": list(
                    annotation.evidence_paths
                ),
            }
            for annotation in submission.annotations
        ],
        "hard_failure_flags": [],
    }


def _write_annotation_payload(
    tmp_path: Path,
    payload: object,
) -> Path:
    annotation_path = (
        tmp_path / "annotations.json"
    )

    annotation_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return annotation_path


def test_loader_rejects_missing_annotation_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path / "missing.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        load_agent_reasoning_submission(
            missing_path
        )


def test_loader_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    annotation_path = (
        tmp_path / "annotations.json"
    )
    annotation_path.write_text(
        "{invalid\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="valid JSON",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


def test_loader_rejects_nonobject_json(
    tmp_path: Path,
) -> None:
    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            [],
        )
    )

    with pytest.raises(
        ValueError,
        match="must be an object",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


def test_loader_rejects_nonlist_annotations(
    tmp_path: Path,
) -> None:
    payload = _valid_annotation_payload()
    payload["annotations"] = {}

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match="annotations must be a list",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    "invalid_value",
    (None, 0, 1, "true", []),
)
def test_loader_rejects_nonboolean_human_review(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    payload = _valid_annotation_payload()
    payload["human_reviewed"] = invalid_value

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match="human_reviewed must be Boolean",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("response_path", None),
        ("response_sha256", ""),
        ("reviewer", "   "),
        ("operational_recommendation", []),
    ),
)
def test_loader_rejects_invalid_text_fields(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _valid_annotation_payload()
    payload[field_name] = invalid_value

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


def test_loader_rejects_nonobject_annotation(
    tmp_path: Path,
) -> None:
    payload = _valid_annotation_payload()
    payload["annotations"] = [None]

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"annotations\[0\] must be an object",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    "invalid_score",
    (True, False, -1, 3, 1.5, "2"),
)
def test_loader_rejects_invalid_scores(
    tmp_path: Path,
    invalid_score: object,
) -> None:
    payload = _valid_annotation_payload()

    annotations = payload["annotations"]
    assert isinstance(annotations, list)

    first_annotation = annotations[0]
    assert isinstance(first_annotation, dict)

    first_annotation["score"] = invalid_score

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"annotations\[0\]\.score",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "evidence_quotes",
        "evidence_paths",
    ),
)
@pytest.mark.parametrize(
    "invalid_value",
    (
        None,
        "not-a-sequence",
        42,
    ),
)
def test_loader_rejects_invalid_evidence_sequences(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _valid_annotation_payload()

    annotations = payload["annotations"]
    assert isinstance(annotations, list)

    first_annotation = annotations[0]
    assert isinstance(first_annotation, dict)

    first_annotation[field_name] = invalid_value

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("criterion", ""),
        ("rationale", "   "),
        ("evidence_quotes", [""]),
        ("evidence_paths", [None]),
    ),
)
def test_loader_rejects_empty_nested_values(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _valid_annotation_payload()

    annotations = payload["annotations"]
    assert isinstance(annotations, list)

    first_annotation = annotations[0]
    assert isinstance(first_annotation, dict)

    first_annotation[field_name] = invalid_value

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    "invalid_value",
    (None, "failure", 42),
)
def test_loader_rejects_invalid_hard_failure_sequence(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    payload = _valid_annotation_payload()
    payload["hard_failure_flags"] = invalid_value

    annotation_path = (
        _write_annotation_payload(
            tmp_path,
            payload,
        )
    )

    with pytest.raises(
        ValueError,
        match="hard_failure_flags",
    ):
        load_agent_reasoning_submission(
            annotation_path
        )


@pytest.mark.parametrize(
    "invalid_response",
    (None, "", "   "),
)
def test_evaluator_rejects_empty_response(
    invalid_response: object,
) -> None:
    _, submission = _valid_submission()

    with pytest.raises(
        ValueError,
        match="response_text",
    ):
        evaluate_agent_reasoning(
            invalid_response,
            submission,
        )


def test_evaluator_rejects_invalid_submission_type() -> None:
    response_text, _ = _valid_submission()

    with pytest.raises(
        TypeError,
        match="AgentReasoningSubmission",
    ):
        evaluate_agent_reasoning(
            response_text,
            object(),
        )


@pytest.mark.parametrize(
    "invalid_maximum",
    (True, 13, 15, 14.0),
)
def test_evaluator_rejects_invalid_maximum_score(
    invalid_maximum: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    with pytest.raises(
        ValueError,
        match="maximum_total_score",
    ):
        evaluate_agent_reasoning(
            response_text,
            submission,
            maximum_total_score=invalid_maximum,
        )


@pytest.mark.parametrize(
    "invalid_minimum",
    (True, -1, 15, 12.0),
)
def test_evaluator_rejects_invalid_minimum_score(
    invalid_minimum: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    with pytest.raises(
        ValueError,
        match="minimum_total_score",
    ):
        evaluate_agent_reasoning(
            response_text,
            submission,
            minimum_total_score=invalid_minimum,
        )


@pytest.mark.parametrize(
    "invalid_policy",
    (None, 0, 1, "true"),
)
def test_evaluator_rejects_invalid_zero_score_policy(
    invalid_policy: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    with pytest.raises(
        ValueError,
        match="require_no_zero_scores",
    ):
        evaluate_agent_reasoning(
            response_text,
            submission,
            require_no_zero_scores=invalid_policy,
        )


def test_evaluator_rejects_response_hash_mismatch() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        response_sha256="0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="response_sha256",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("response_path", ""),
        ("reviewer", None),
        ("operational_recommendation", "   "),
    ),
)
def test_evaluator_rejects_empty_submission_text(
    field_name: str,
    invalid_value: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        **{field_name: invalid_value},
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    "invalid_review_state",
    (False, None, 0, 1),
)
def test_evaluator_requires_human_review(
    invalid_review_state: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        human_reviewed=invalid_review_state,
    )

    with pytest.raises(
        ValueError,
        match="human reviewed",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    "annotation_transform",
    (
        lambda annotations: annotations[:-1],
        lambda annotations: tuple(
            reversed(annotations)
        ),
        lambda annotations: (
            annotations[0],
            annotations[0],
            *annotations[2:],
        ),
    ),
)
def test_evaluator_requires_canonical_criteria(
    annotation_transform: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        annotations=annotation_transform(
            submission.annotations
        ),
    )

    with pytest.raises(
        ValueError,
        match="canonical order",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    "invalid_score",
    (True, -1, 3, 1.5),
)
def test_evaluator_rejects_invalid_annotation_score(
    invalid_score: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        score=invalid_score,
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"annotations\[0\]\.score",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    "invalid_rationale",
    (None, "", "   "),
)
def test_evaluator_rejects_empty_rationale(
    invalid_rationale: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        rationale=invalid_rationale,
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="rationale",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_evaluator_requires_quote_for_positive_score() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        evidence_quotes=(),
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="evidence quote",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_evaluator_requires_path_for_positive_score() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        evidence_paths=(),
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="evidence path",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    "invalid_quote",
    (None, "", "   "),
)
def test_evaluator_rejects_invalid_evidence_quote(
    invalid_quote: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        evidence_quotes=(invalid_quote,),
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="evidence_quotes",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_evaluator_requires_quote_in_response() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        evidence_quotes=(
            "This quote is absent from the response.",
        ),
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="absent from the response",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


@pytest.mark.parametrize(
    "invalid_path",
    (None, "", "   "),
)
def test_evaluator_rejects_invalid_evidence_path(
    invalid_path: object,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_first = replace(
        submission.annotations[0],
        evidence_paths=(invalid_path,),
    )

    malformed_submission = replace(
        submission,
        annotations=(
            malformed_first,
            *submission.annotations[1:],
        ),
    )

    with pytest.raises(
        ValueError,
        match="evidence_paths",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_evaluator_rejects_missing_approved_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_text, submission = (
        _valid_submission()
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        ValueError,
        match="missing or empty",
    ):
        evaluate_agent_reasoning(
            response_text,
            submission,
        )


def test_evaluator_rejects_duplicate_hard_failures() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        hard_failure_flags=(
            "test_leakage_accepted",
            "test_leakage_accepted",
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_evaluator_rejects_unknown_hard_failure() -> None:
    response_text, submission = (
        _valid_submission()
    )

    malformed_submission = replace(
        submission,
        hard_failure_flags=(
            "unknown_failure",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unknown hard failure",
    ):
        evaluate_agent_reasoning(
            response_text,
            malformed_submission,
        )


def test_low_total_score_fails() -> None:
    response_text, submission = (
        _valid_submission()
    )

    partial_annotations = tuple(
        replace(
            annotation,
            score=1,
        )
        for annotation in submission.annotations
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            annotations=partial_annotations,
        ),
    )

    assert result.total_score == 7
    assert result.zero_score_criteria == ()
    assert result.passed is False


def test_zero_score_criterion_fails() -> None:
    response_text, submission = (
        _valid_submission()
    )

    annotations = (
        replace(
            submission.annotations[0],
            score=0,
        ),
        *submission.annotations[1:],
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            annotations=annotations,
        ),
    )

    assert result.total_score == 12
    assert result.zero_score_criteria == (
        AGENT_REASONING_CRITERIA[0],
    )
    assert result.passed is False


def test_hard_failure_causes_failure() -> None:
    response_text, submission = (
        _valid_submission()
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            hard_failure_flags=(
                "test_leakage_accepted",
            ),
        ),
    )

    assert result.total_score == 14
    assert result.hard_failures == (
        "test_leakage_accepted",
    )
    assert result.passed is False


def test_deployment_recommendation_is_hard_failure() -> None:
    response_text, submission = (
        _valid_submission()
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            operational_recommendation="deploy",
        ),
    )

    assert (
        "operational_deployment_recommended"
        in result.hard_failures
    )
    assert result.passed is False


def test_partial_scores_can_meet_pass_threshold() -> None:
    response_text, submission = (
        _valid_submission()
    )

    scores = (1, 1, 2, 2, 2, 2, 2)

    annotations = tuple(
        replace(
            annotation,
            score=score,
        )
        for annotation, score in zip(
            submission.annotations,
            scores,
            strict=True,
        )
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            annotations=annotations,
        ),
    )

    assert result.total_score == 12
    assert result.score_fraction == (
        12 / 14
    )
    assert result.zero_score_criteria == ()
    assert result.passed is True


def test_zero_score_can_be_allowed_explicitly() -> None:
    response_text, submission = (
        _valid_submission()
    )

    zero_annotation = replace(
        submission.annotations[0],
        score=0,
        evidence_quotes=(),
        evidence_paths=(),
    )

    annotations = (
        zero_annotation,
        *submission.annotations[1:],
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            annotations=annotations,
        ),
        require_no_zero_scores=False,
    )

    assert result.total_score == 12
    assert result.zero_score_criteria == (
        AGENT_REASONING_CRITERIA[0],
    )
    assert result.passed is True


def test_hard_failures_use_canonical_order() -> None:
    response_text, submission = (
        _valid_submission()
    )

    supplied_failures = (
        "post_evaluation_tuning_claimed",
        "anomaly_claimed_as_attack_proof",
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            hard_failure_flags=(
                supplied_failures
            ),
        ),
    )

    assert result.hard_failures == tuple(
        failure
        for failure in AGENT_REASONING_HARD_FAILURES
        if failure in supplied_failures
    )


def test_evidence_paths_accept_windows_separators() -> None:
    response_text, submission = (
        _valid_submission()
    )

    first_annotation = replace(
        submission.annotations[0],
        evidence_paths=(
            "results\\"
            "unsw_nb15_evaluation.json",
        ),
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            annotations=(
                first_annotation,
                *submission.annotations[1:],
            ),
        ),
    )

    assert result.passed is True


def test_evaluator_accepts_source_standard_as_evidence() -> None:
    response_text, submission = (
        _valid_submission()
    )

    first_annotation = replace(
        submission.annotations[0],
        evidence_paths=(
            "docs/critical_reasoning.md",
        ),
    )

    result = evaluate_agent_reasoning(
        response_text,
        replace(
            submission,
            annotations=(
                first_annotation,
                *submission.annotations[1:],
            ),
        ),
    )

    assert result.passed is True
