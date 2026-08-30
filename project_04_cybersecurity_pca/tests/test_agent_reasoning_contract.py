from pathlib import Path

import yaml


EXPECTED_REASONING_CRITERIA = (
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


def test_phase_nine_reasoning_configuration_contract() -> None:
    configuration = yaml.safe_load(
        Path("configs/baseline.yaml").read_text(
            encoding="utf-8"
        )
    )

    contract = configuration[
        "agent_reasoning_evaluation"
    ]

    assert contract["phase"] == 9
    assert contract["rubric_version"] == "1.0.0"

    assert (
        contract["evaluation_subject"]
        == "phase_08_unsw_nb15_reasoning_response"
    )

    assert (
        contract["response_path"]
        == "results/"
        "phase_09_agent_reasoning_response.md"
    )

    assert (
        contract["annotation_path"]
        == "results/"
        "phase_09_agent_reasoning_annotations.json"
    )

    assert (
        contract["source_standard_path"]
        == "docs/critical_reasoning.md"
    )

    assert (
        contract["automated_free_text_scoring"]
        is False
    )
    assert contract["human_review_required"] is True

    assert (
        contract["frozen_phase_08_commit"]
        == "c3867c45ee910d68648329ee8090d990169a60f4"
    )

    assert tuple(
        contract["evidence_paths"]
    ) == (
        "results/unsw_nb15_evaluation.json",
        "reports/tables/unsw_nb15_metrics.csv",
        (
            "reports/tables/"
            "unsw_nb15_attack_category_metrics.csv"
        ),
        "agent_trace/phase_08.md",
    )

    for evidence_path in contract["evidence_paths"]:
        path = Path(evidence_path)
        assert path.is_file()
        assert path.stat().st_size > 0

    assert contract["score_levels"] == {
        "missing_or_incorrect": 0,
        "partial": 1,
        "complete_and_evidence_grounded": 2,
    }

    assert tuple(contract["criteria"]) == (
        EXPECTED_REASONING_CRITERIA
    )

    assert contract["minimum_total_score"] == 12
    assert contract["maximum_total_score"] == 14
    assert contract["require_no_zero_scores"] is True

    assert tuple(
        contract["hard_failure_rules"]
    ) == EXPECTED_HARD_FAILURES

    assert (
        contract[
            "expected_operational_recommendation"
        ]
        == "not_recommended"
    )

    assert contract["output_paths"] == {
        "summary_json": (
            "results/"
            "agent_reasoning_evaluation.json"
        ),
        "rubric_csv": (
            "reports/tables/"
            "agent_reasoning_rubric.csv"
        ),
        "scores_figure": (
            "reports/figures/"
            "agent_reasoning_scores.png"
        ),
    }
