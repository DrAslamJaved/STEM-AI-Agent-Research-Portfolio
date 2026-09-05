"""Phase 7 final-evaluation tests: config safety, trace freezing, bootstrap."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from evidence_agent.audit.bootstrap import (
    BootstrapError,
    build_claim_outcomes,
    paired_bootstrap_confidence_intervals,
)
from evidence_agent.cli import main
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.data.schemas import AuditDecision, Citation, Verdict
from evidence_agent.evaluation import final as final_module
from evidence_agent.evaluation.final import FinalEvaluationError, run_evaluate_command
from evidence_agent.evaluation.final_config import (
    CONFIG_SCHEMA_VERSION,
    EVALUATION_LABEL,
    FinalEvaluationConfigError,
    load_final_evaluation_config,
)
from evidence_agent.verification.agent import VerificationTrace
from evidence_agent.verification.scifact import GoldClaimAnnotation
from tests.helpers import write_citation_audit_scifact_dataset


PLACEHOLDER_SHA256 = "0" * 64


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _minimal_output_block(tmp_path: Path) -> dict:
    return {
        "result_path": "../results/final_evaluation_dev.json",
        "trace_path": "../artifacts/final_evaluation_dev_trace.json",
        "report_path": "../reports/phase_07_final_evaluation.md",
        "agent_trace_path": "../agent_trace/phase_07_final_evaluation.md",
    }


def _placeholder_config_payload(tmp_path: Path) -> dict:
    """A schema-valid config whose artifact hashes are never actually checked
    by ``load_final_evaluation_config`` (only ``run_final_evaluation`` hashes
    files), so dummy paths and a placeholder digest are sufficient here."""
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
        "bootstrap": {"enabled": False, "resamples": 10, "seed": 1, "confidence_level": 0.9},
        "output": _minimal_output_block(tmp_path),
    }


def test_load_final_evaluation_config_resolves_paths_relative_to_the_config_file(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    payload = _placeholder_config_payload(tmp_path)
    config_path = config_dir / "final.yaml"
    _write_yaml(config_path, payload)

    config = load_final_evaluation_config(config_path)

    assert config.output.result_path == (tmp_path / "results" / "final_evaluation_dev.json").resolve()
    assert config.output.trace_path == (
        tmp_path / "artifacts" / "final_evaluation_dev_trace.json"
    ).resolve()
    assert config.output.report_path == (
        tmp_path / "reports" / "phase_07_final_evaluation.md"
    ).resolve()
    assert config.output.agent_trace_path == (
        tmp_path / "agent_trace" / "phase_07_final_evaluation.md"
    ).resolve()


def test_load_final_evaluation_config_rejects_an_unsupported_schema_version(tmp_path: Path) -> None:
    payload = _placeholder_config_payload(tmp_path)
    payload["schema_version"] = "not_a_real_schema"
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(FinalEvaluationConfigError, match="schema_version"):
        load_final_evaluation_config(config_path)


def test_load_final_evaluation_config_rejects_a_wrong_label(tmp_path: Path) -> None:
    payload = _placeholder_config_payload(tmp_path)
    payload["label"] = "independent_test"
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(FinalEvaluationConfigError, match="label"):
        load_final_evaluation_config(config_path)


@pytest.mark.parametrize(
    "forbidden_relative_path",
    ["../results/citation_audit_dev.json", "../results/citation_audit_cross_validation.json"],
)
def test_load_final_evaluation_config_refuses_to_target_a_phase_06_result_file(
    tmp_path: Path, forbidden_relative_path: str
) -> None:
    payload = _placeholder_config_payload(tmp_path)
    payload["output"]["result_path"] = forbidden_relative_path
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(FinalEvaluationConfigError, match="Phase 6 result path"):
        load_final_evaluation_config(config_path)


def _prepare_frozen_artifacts(tmp_path: Path) -> dict[str, Path]:
    """Build tiny, real Phase 05/06 artifacts once, using the existing CLI."""
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


def _final_config_payload(paths: dict[str, Path], **overrides: object) -> dict:
    payload = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "label": EVALUATION_LABEL,
        "artifacts": {
            name: {"path": str(paths[name]), "sha256": sha256_file(paths[name])}
            for name in ("corpus", "claims_dev", "bm25_index", "verifier_model", "calibration_report")
        },
        "runtime": {"retrieval_k": 2},
        "bootstrap": {"enabled": True, "resamples": 25, "seed": 1234, "confidence_level": 0.8},
        "output": {
            "result_path": "../results/final_evaluation_dev.json",
            "trace_path": "../artifacts/final_evaluation_dev_trace.json",
            "report_path": "../reports/phase_07_final_evaluation.md",
            "agent_trace_path": "../agent_trace/phase_07_final_evaluation.md",
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "artifact_name",
    ["corpus", "claims_dev", "bm25_index", "verifier_model", "calibration_report"],
)
def test_run_evaluate_command_rejects_a_sha256_mismatch_for_each_required_artifact(
    tmp_path: Path, artifact_name: str
) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    payload = _final_config_payload(paths)
    payload["artifacts"][artifact_name]["sha256"] = PLACEHOLDER_SHA256
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(FinalEvaluationError, match=artifact_name):
        run_evaluate_command(config_path)


def test_run_evaluate_command_does_not_load_gold_before_the_trace_file_exists(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, _final_config_payload(paths))
    config = load_final_evaluation_config(config_path)

    real_loader = final_module.load_gold_claim_annotations
    calls: list[bool] = []

    def _checked_loader(*args: object, **kwargs: object):
        calls.append(config.output.trace_path.is_file())
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(final_module, "load_gold_claim_annotations", _checked_loader)

    report = final_module.run_final_evaluation(config)

    assert calls == [True]
    assert report["trace_artifact"]["sha256"] == sha256_file(config.output.trace_path)


def test_run_evaluate_command_applies_both_policies_to_the_identical_claim_order(
    tmp_path: Path,
) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, _final_config_payload(paths))

    summary = run_evaluate_command(config_path)
    report = json.loads(Path(summary["result_path"]).read_text(encoding="utf-8"))

    # trace_artifact.path is recorded relative to the project root (tmp_path
    # here), not the process working directory, so it is rejoined with the
    # project root before being opened.
    raw_trace = json.loads((tmp_path / report["trace_artifact"]["path"]).read_text(encoding="utf-8"))
    raw_claim_order = [trace["decision"]["claim_id"] for trace in raw_trace["traces"]]
    selected_order = [decision["claim_id"] for decision in report["selected_decisions"]]
    phase_05_order = [decision["claim_id"] for decision in report["phase_05_decisions"]]

    assert selected_order == phase_05_order == raw_claim_order
    assert report["schema_version"] == "evidence_agent_final_evaluation_v1"
    assert report["evaluation_label"] == "held_out_development_evaluation"
    assert report["is_independent_test"] is False


_DRIVE_LETTER_PATTERN = re.compile(r"^[A-Za-z]:")


def _assert_project_relative_posix_path(path_str: str) -> None:
    assert isinstance(path_str, str)
    assert "\\" not in path_str, f"expected forward slashes, got {path_str!r}"
    assert not _DRIVE_LETTER_PATTERN.match(path_str), f"expected no drive letter, got {path_str!r}"
    assert not Path(path_str).is_absolute(), f"expected a relative path, got {path_str!r}"


def test_run_evaluate_command_records_project_relative_posix_paths(tmp_path: Path) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, _final_config_payload(paths))

    summary = run_evaluate_command(config_path)
    report = json.loads(Path(summary["result_path"]).read_text(encoding="utf-8"))

    _assert_project_relative_posix_path(report["config_path"])
    for artifact_name in ("corpus", "claims_dev", "bm25_index", "verifier_model", "calibration_report"):
        _assert_project_relative_posix_path(report["artifacts"][artifact_name]["path"])
    _assert_project_relative_posix_path(report["trace_artifact"]["path"])
    for output_field in ("agent_trace_path", "report_path", "result_path", "trace_path"):
        _assert_project_relative_posix_path(report["output"][output_field])

    # The frozen trace file is still reachable by joining the recorded
    # project-relative path back onto the config's own project root.
    trace_path = tmp_path / report["trace_artifact"]["path"]
    assert trace_path.is_file()
    assert sha256_file(trace_path) == report["trace_artifact"]["sha256"]


def test_run_evaluate_command_never_writes_to_a_phase_06_result_path(tmp_path: Path) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    payload = _final_config_payload(paths)
    payload["output"]["result_path"] = "../results/citation_audit_dev.json"
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, payload)

    with pytest.raises(FinalEvaluationConfigError, match="Phase 6 result path"):
        run_evaluate_command(config_path)


def test_cli_evaluate_command_runs_the_final_pipeline_end_to_end(tmp_path: Path, capsys) -> None:
    paths = _prepare_frozen_artifacts(tmp_path)
    capsys.readouterr()
    config_path = tmp_path / "configs" / "final.yaml"
    _write_yaml(config_path, _final_config_payload(paths))

    assert main(["evaluate", "--config", str(config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)

    result_path = Path(printed["result_path"])
    assert result_path.name == "final_evaluation_dev.json"
    assert result_path.is_file()
    assert printed["evaluation_label"] == "held_out_development_evaluation"

    bootstrap = printed["bootstrap_confidence_intervals"]
    assert bootstrap["claim_count"] == 1
    assert bootstrap["resamples"] == 25
    for metrics in bootstrap["metrics"].values():
        assert metrics["lower"] <= metrics["upper"]
    expected_metrics = {
        "citation_correctness_f1",
        "claim_macro_f1",
        "coverage",
        "evidence_sentence_f1",
        "faithfulness",
        "unsupported_assertion_rate",
    }
    assert set(bootstrap["metrics"]) == expected_metrics


def _bootstrap_fixture() -> tuple[list, dict[int, GoldClaimAnnotation]]:
    gold_annotations = {
        1: GoldClaimAnnotation(
            claim_id=1,
            verdict=Verdict.SUPPORT,
            citations=(Citation(doc_id=10, sentence_ids=(0,), stance=Verdict.SUPPORT),),
        ),
        2: GoldClaimAnnotation(
            claim_id=2,
            verdict=Verdict.CONTRADICT,
            citations=(Citation(doc_id=11, sentence_ids=(0,), stance=Verdict.CONTRADICT),),
        ),
        3: GoldClaimAnnotation(claim_id=3, verdict=Verdict.NO_EVIDENCE, citations=()),
        4: GoldClaimAnnotation(
            claim_id=4,
            verdict=Verdict.SUPPORT,
            citations=(Citation(doc_id=12, sentence_ids=(0,), stance=Verdict.SUPPORT),),
        ),
    }
    selected_traces = [
        VerificationTrace(
            decision=AuditDecision(
                claim_id=1,
                verdict=Verdict.SUPPORT,
                confidence=0.9,
                citations=(Citation(doc_id=10, sentence_ids=(0,), stance=Verdict.SUPPORT),),
            ),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(claim_id=2, verdict=Verdict.NO_EVIDENCE, confidence=0.4),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(claim_id=3, verdict=Verdict.NO_EVIDENCE, confidence=0.6),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(
                claim_id=4,
                verdict=Verdict.SUPPORT,
                confidence=0.7,
                citations=(Citation(doc_id=12, sentence_ids=(0,), stance=Verdict.SUPPORT),),
            ),
            candidates=(),
        ),
    ]
    baseline_traces = [
        VerificationTrace(
            decision=AuditDecision(
                claim_id=1,
                verdict=Verdict.SUPPORT,
                confidence=0.6,
                citations=(Citation(doc_id=10, sentence_ids=(0,), stance=Verdict.SUPPORT),),
            ),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(
                claim_id=2,
                verdict=Verdict.CONTRADICT,
                confidence=0.55,
                citations=(Citation(doc_id=11, sentence_ids=(0,), stance=Verdict.CONTRADICT),),
            ),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(
                claim_id=3,
                verdict=Verdict.SUPPORT,
                confidence=0.51,
                citations=(Citation(doc_id=12, sentence_ids=(0,), stance=Verdict.SUPPORT),),
            ),
            candidates=(),
        ),
        VerificationTrace(
            decision=AuditDecision(claim_id=4, verdict=Verdict.NO_EVIDENCE, confidence=0.3),
            candidates=(),
        ),
    ]
    return selected_traces, baseline_traces, gold_annotations


def test_paired_bootstrap_confidence_intervals_are_deterministic_and_well_formed() -> None:
    selected_traces, baseline_traces, gold_annotations = _bootstrap_fixture()
    selected_outcomes = build_claim_outcomes(selected_traces, gold_annotations)
    baseline_outcomes = build_claim_outcomes(baseline_traces, gold_annotations)

    first = paired_bootstrap_confidence_intervals(
        selected_outcomes, baseline_outcomes, resamples=200, seed=42, confidence_level=0.9
    )
    second = paired_bootstrap_confidence_intervals(
        selected_outcomes, baseline_outcomes, resamples=200, seed=42, confidence_level=0.9
    )

    assert first == second
    assert first["claim_count"] == 4
    expected_metrics = {
        "citation_correctness_f1",
        "claim_macro_f1",
        "coverage",
        "evidence_sentence_f1",
        "faithfulness",
        "unsupported_assertion_rate",
    }
    assert set(first["metrics"]) == expected_metrics
    for metric_bounds in first["metrics"].values():
        assert metric_bounds["lower"] <= metric_bounds["mean"] <= metric_bounds["upper"]


def test_paired_bootstrap_confidence_intervals_reject_mismatched_claim_populations() -> None:
    selected_traces, baseline_traces, gold_annotations = _bootstrap_fixture()
    selected_outcomes = build_claim_outcomes(selected_traces, gold_annotations)
    baseline_outcomes = build_claim_outcomes(baseline_traces[:-1], {1: gold_annotations[1], 2: gold_annotations[2], 3: gold_annotations[3]})

    with pytest.raises(BootstrapError, match="identical claim IDs"):
        paired_bootstrap_confidence_intervals(
            selected_outcomes, baseline_outcomes, resamples=10, seed=1, confidence_level=0.9
        )
