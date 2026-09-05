"""Phase 8 tests for the controlled direct-RAG versus audited-agent study."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from evidence_agent.cli import main
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.data.schemas import AuditDecision, Citation, Verdict
from evidence_agent.evaluation import controlled as controlled_module
from evidence_agent.evaluation.controlled import (
    ControlledExperimentsError,
    evaluate_official_scifact_traces,
    official_predictions_from_traces,
    paired_bootstrap_official_confidence_intervals,
    run_adversarial_evaluator_suite,
    run_controlled_experiments_command,
)
from evidence_agent.evaluation.controlled_config import (
    CONFIG_SCHEMA_VERSION,
    EVALUATION_LABEL,
    ControlledExperimentsConfigError,
    load_controlled_experiments_config,
)
from evidence_agent.verification.agent import VerificationTrace
from evidence_agent.verification.scifact import GoldClaimAnnotation
from tests.helpers import write_citation_audit_scifact_dataset


PLACEHOLDER_SHA256 = "0" * 64
_DRIVE_LETTER_PATTERN = re.compile(r"^[A-Za-z]:")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _output_block() -> dict[str, str]:
    return {
        "result_path": "../results/controlled_experiments_dev.json",
        "trace_path": "../artifacts/controlled_experiments_dev_trace.json",
        "direct_predictions_path": "../artifacts/controlled_experiments_direct_rag_predictions.jsonl",
        "audited_predictions_path": "../artifacts/controlled_experiments_audited_agent_predictions.jsonl",
        "report_path": "../reports/phase_08_controlled_experiments.md",
        "agent_trace_path": "../agent_trace/phase_08_controlled_experiments.md",
    }


def _placeholder_config_payload(tmp_path: Path) -> dict[str, object]:
    dummy = tmp_path / "dummy.txt"
    dummy.write_text("placeholder", encoding="utf-8")
    artifact = {"path": str(dummy), "sha256": PLACEHOLDER_SHA256}
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "label": EVALUATION_LABEL,
        "artifacts": {
            "corpus": artifact,
            "claims_dev": artifact,
            "bm25_index": artifact,
            "verifier_model": artifact,
            "calibration_report": artifact,
        },
        "runtime": {"retrieval_k": 2},
        "direct_rag": {"max_sentences_per_citation": 3},
        "bootstrap": {"enabled": False, "resamples": 10, "seed": 1, "confidence_level": 0.9},
        "output": _output_block(),
    }


def test_load_controlled_config_resolves_paths_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, _placeholder_config_payload(tmp_path))

    config = load_controlled_experiments_config(config_path)

    assert config.output.result_path == (tmp_path / "results" / "controlled_experiments_dev.json").resolve()
    assert config.output.trace_path == (
        tmp_path / "artifacts" / "controlled_experiments_dev_trace.json"
    ).resolve()
    assert config.output.direct_predictions_path == (
        tmp_path / "artifacts" / "controlled_experiments_direct_rag_predictions.jsonl"
    ).resolve()
    assert config.output.audited_predictions_path == (
        tmp_path / "artifacts" / "controlled_experiments_audited_agent_predictions.jsonl"
    ).resolve()


def test_load_controlled_config_rejects_unknown_schema_version(tmp_path: Path) -> None:
    payload = _placeholder_config_payload(tmp_path)
    payload["schema_version"] = "not-a-schema"
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(ControlledExperimentsConfigError, match="schema_version"):
        load_controlled_experiments_config(config_path)


@pytest.mark.parametrize(
    ("output_field", "forbidden_path"),
    [
        ("result_path", "../results/citation_audit_dev.json"),
        ("result_path", "../results/final_evaluation_dev.json"),
        ("report_path", "../reports/phase_07_final_evaluation.md"),
    ],
)
def test_load_controlled_config_refuses_to_overwrite_prior_phase_outputs(
    tmp_path: Path, output_field: str, forbidden_path: str
) -> None:
    payload = _placeholder_config_payload(tmp_path)
    output = payload["output"]
    assert isinstance(output, dict)
    output[output_field] = forbidden_path
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(ControlledExperimentsConfigError, match="Phase 6 or Phase 7 output"):
        load_controlled_experiments_config(config_path)


def test_load_controlled_config_refuses_to_overwrite_a_declared_input(tmp_path: Path) -> None:
    payload = _placeholder_config_payload(tmp_path)
    artifact_paths = payload["artifacts"]
    assert isinstance(artifact_paths, dict)
    corpus = artifact_paths["corpus"]
    assert isinstance(corpus, dict)
    output = payload["output"]
    assert isinstance(output, dict)
    output["trace_path"] = corpus["path"]
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(ControlledExperimentsConfigError, match="declared input artifact"):
        load_controlled_experiments_config(config_path)


def _prepare_frozen_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Build small, real Phase 05/06 artifacts once for a Phase 8 test."""
    dataset = write_citation_audit_scifact_dataset(tmp_path / "scifact")
    index_path = tmp_path / "artifacts" / "bm25.json"
    model_path = tmp_path / "artifacts" / "verifier.joblib"
    calibration_path = tmp_path / "results" / "calibration.json"

    assert main(
        [
            "build-index",
            "--corpus-path",
            str(dataset / "corpus.jsonl"),
            "--index-path",
            str(index_path),
        ]
    ) == 0
    assert main(
        [
            "calibrate-citation-audit",
            "--corpus-path",
            str(dataset / "corpus.jsonl"),
            "--train-claims-path",
            str(dataset / "claims_train.jsonl"),
            "--development-claims-path",
            str(dataset / "claims_dev.jsonl"),
            "--cross-validation-dir",
            str(dataset / "cross_validation"),
            "--index-path",
            str(index_path),
            "--artifact-dir",
            str(tmp_path / "artifacts" / "audit_cv"),
            "--report-path",
            str(calibration_path),
            "--assertion-thresholds",
            "0",
            "--sentence-thresholds",
            "0",
            "--max-sentences-per-citation",
            "1",
            "--minimum-coverage",
            "0",
            "--max-features",
            "100",
            "--retrieval-k",
            "2",
        ]
    ) == 0
    assert main(
        [
            "train-verifier",
            "--corpus-path",
            str(dataset / "corpus.jsonl"),
            "--train-claims-path",
            str(dataset / "claims_train.jsonl"),
            "--model-path",
            str(model_path),
            "--max-features",
            "100",
        ]
    ) == 0

    return {
        "corpus": dataset / "corpus.jsonl",
        "claims_dev": dataset / "claims_dev.jsonl",
        "train_claims": dataset / "claims_train.jsonl",
        "bm25_index": index_path,
        "verifier_model": model_path,
        "calibration_report": calibration_path,
    }


def _controlled_config_payload(paths: dict[str, Path], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "label": EVALUATION_LABEL,
        "artifacts": {
            name: {"path": str(paths[name]), "sha256": sha256_file(paths[name])}
            for name in (
                "corpus",
                "claims_dev",
                "train_claims",
                "bm25_index",
                "verifier_model",
                "calibration_report",
            )
        },
        "runtime": {"retrieval_k": 2},
        "direct_rag": {"max_sentences_per_citation": 3},
        "bootstrap": {"enabled": True, "resamples": 25, "seed": 1234, "confidence_level": 0.8},
        "output": _output_block(),
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "artifact_name",
    ["corpus", "claims_dev", "bm25_index", "verifier_model", "calibration_report"],
)
def test_controlled_experiments_rejects_hash_mismatch_for_required_inputs(
    tmp_path: Path, artifact_name: str
) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    payload = _controlled_config_payload(paths)
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifact = artifacts[artifact_name]
    assert isinstance(artifact, dict)
    artifact["sha256"] = PLACEHOLDER_SHA256
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(ControlledExperimentsError, match=artifact_name):
        run_controlled_experiments_command(config_path)


def _trace_with_citation(claim_id: int, citation: Citation | None) -> VerificationTrace:
    if citation is None:
        return VerificationTrace(
            decision=AuditDecision(claim_id, Verdict.NO_EVIDENCE, 1.0), candidates=()
        )
    return VerificationTrace(
        decision=AuditDecision(claim_id, citation.stance, 0.9, (citation,)), candidates=()
    )


def test_official_scifact_evaluation_requires_document_stance_and_complete_rationale() -> None:
    gold = {
        1: GoldClaimAnnotation(
            claim_id=1,
            verdict=Verdict.SUPPORT,
            citations=(Citation(doc_id=10, sentence_ids=(0, 1), stance=Verdict.SUPPORT),),
        )
    }
    exact = _trace_with_citation(
        1, Citation(doc_id=10, sentence_ids=(0, 1), stance=Verdict.SUPPORT)
    )
    incomplete = _trace_with_citation(
        1, Citation(doc_id=10, sentence_ids=(0,), stance=Verdict.SUPPORT)
    )
    wrong_document = _trace_with_citation(
        1, Citation(doc_id=11, sentence_ids=(0, 1), stance=Verdict.SUPPORT)
    )
    wrong_stance = _trace_with_citation(
        1, Citation(doc_id=10, sentence_ids=(0, 1), stance=Verdict.CONTRADICT)
    )

    assert evaluate_official_scifact_traces((exact,), gold).summary_dict()["abstract_level"]["f1"] == 1.0
    for trace in (incomplete, wrong_document, wrong_stance):
        summary = evaluate_official_scifact_traces((trace,), gold).summary_dict()
        assert summary["abstract_level"]["f1"] == 0.0
        assert summary["sentence_level"]["f1"] == 0.0


def test_official_predictions_use_the_document_keyed_scifact_submission_shape() -> None:
    trace = _trace_with_citation(
        12, Citation(doc_id=42, sentence_ids=(0, 2), stance=Verdict.SUPPORT)
    )

    assert official_predictions_from_traces((trace,)) == (
        {
            "id": 12,
            "evidence": {"42": {"label": "SUPPORT", "sentences": [0, 2]}},
        },
    )


def _official_bootstrap_fixture() -> tuple[list[VerificationTrace], list[VerificationTrace], dict[int, GoldClaimAnnotation]]:
    gold = {
        1: GoldClaimAnnotation(1, Verdict.SUPPORT, (Citation(10, (0,), Verdict.SUPPORT),)),
        2: GoldClaimAnnotation(2, Verdict.CONTRADICT, (Citation(11, (0,), Verdict.CONTRADICT),)),
        3: GoldClaimAnnotation(3, Verdict.NO_EVIDENCE, ()),
        4: GoldClaimAnnotation(4, Verdict.SUPPORT, (Citation(12, (0,), Verdict.SUPPORT),)),
    }
    audited = [
        _trace_with_citation(1, Citation(10, (0,), Verdict.SUPPORT)),
        _trace_with_citation(2, None),
        _trace_with_citation(3, None),
        _trace_with_citation(4, Citation(12, (0,), Verdict.SUPPORT)),
    ]
    direct = [
        _trace_with_citation(1, Citation(10, (0,), Verdict.SUPPORT)),
        _trace_with_citation(2, Citation(11, (0,), Verdict.CONTRADICT)),
        _trace_with_citation(3, Citation(12, (0,), Verdict.SUPPORT)),
        _trace_with_citation(4, None),
    ]
    return audited, direct, gold


def test_official_paired_bootstrap_is_deterministic_and_occurrence_aware() -> None:
    audited, direct, gold = _official_bootstrap_fixture()

    first = paired_bootstrap_official_confidence_intervals(
        audited, direct, gold, resamples=200, seed=42, confidence_level=0.9
    )
    second = paired_bootstrap_official_confidence_intervals(
        audited, direct, gold, resamples=200, seed=42, confidence_level=0.9
    )

    assert first == second
    assert first["claim_count"] == 4
    assert set(first["metrics"]) == {"abstract_level_f1", "sentence_level_f1"}
    for bounds in first["metrics"].values():
        assert bounds["lower"] <= bounds["upper"]


def test_adversarial_evaluator_suite_rejects_every_attack() -> None:
    result = run_adversarial_evaluator_suite()

    assert result["all_passed"] is True
    assert set(result["cases"]) == {"wrong_document", "wrong_stance", "incomplete_rationale"}


def test_controlled_experiments_freezes_trace_and_both_prediction_arms_before_gold(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, _controlled_config_payload(paths))
    config = load_controlled_experiments_config(config_path)
    real_loader = controlled_module.load_gold_claim_annotations
    observed_outputs: list[tuple[bool, bool, bool]] = []

    def _checked_loader(*args: object, **kwargs: object):
        observed_outputs.append(
            (
                config.output.trace_path.is_file(),
                config.output.direct_predictions_path.is_file(),
                config.output.audited_predictions_path.is_file(),
            )
        )
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(controlled_module, "load_gold_claim_annotations", _checked_loader)
    report = controlled_module.run_controlled_experiments(config)

    assert observed_outputs == [(True, True, True)]
    assert report["trace_artifact"]["sha256"] == sha256_file(config.output.trace_path)


def _assert_project_relative_posix_path(path_str: object) -> None:
    assert isinstance(path_str, str)
    assert "\\" not in path_str
    assert not _DRIVE_LETTER_PATTERN.match(path_str)
    assert not Path(path_str).is_absolute()


def test_cli_controlled_experiments_runs_end_to_end_with_matching_arm_identity(
    tmp_path: Path, capsys
) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    capsys.readouterr()
    config_path = tmp_path / "configs" / "controlled.yaml"
    _write_yaml(config_path, _controlled_config_payload(paths))

    assert main(["controlled-experiments", "--config", str(config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    report = json.loads(Path(printed["result_path"]).read_text(encoding="utf-8"))

    assert printed["evaluation_label"] == EVALUATION_LABEL
    assert report["schema_version"] == "evidence_agent_controlled_experiments_v1"
    assert report["is_independent_test"] is False
    assert report["adversarial_evaluator_suite"]["all_passed"] is True
    assert set(report["official_bootstrap_confidence_intervals"]["metrics"]) == {
        "abstract_level_f1",
        "sentence_level_f1",
    }

    trace = json.loads((tmp_path / report["trace_artifact"]["path"]).read_text(encoding="utf-8"))
    direct_predictions_path = tmp_path / report["direct_rag"]["predictions_artifact"]["path"]
    audited_predictions_path = tmp_path / report["audited_agent"]["predictions_artifact"]["path"]
    direct_claim_order = [json.loads(line)["id"] for line in direct_predictions_path.read_text(encoding="utf-8").splitlines()]
    audited_claim_order = [json.loads(line)["id"] for line in audited_predictions_path.read_text(encoding="utf-8").splitlines()]
    raw_claim_order = [record["decision"]["claim_id"] for record in trace["traces"]]
    assert direct_claim_order == audited_claim_order == raw_claim_order

    _assert_project_relative_posix_path(report["config_path"])
    _assert_project_relative_posix_path(report["trace_artifact"]["path"])
    for artifact in report["artifacts"].values():
        _assert_project_relative_posix_path(artifact["path"])
    assert sha256_file(tmp_path / report["trace_artifact"]["path"]) == report["trace_artifact"]["sha256"]

    report_path = tmp_path / report["output"]["report_path"]
    result_sha256 = sha256_file(Path(printed["result_path"]))
    assert result_sha256 in report_path.read_text(encoding="utf-8")


def test_controlled_experiments_command_requires_config() -> None:
    with pytest.raises(ValueError, match="--config is required"):
        main(["controlled-experiments"])
