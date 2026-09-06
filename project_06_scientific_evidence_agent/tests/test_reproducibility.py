"""Phase 9 tests for the clean-environment reproducibility gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evidence_agent.cli import main
from evidence_agent.reproducibility import (
    CONFIG_SCHEMA_VERSION,
    GATE_LABEL,
    MANIFEST_FILENAME,
    ReproducibilityConfigError,
    ReproducibilityError,
    _is_forbidden_committed_path,
    load_reproducibility_config,
    run_reproducibility_check,
    run_reproducibility_command,
    sha256_frozen_result_json,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config_payload(project_root: Path) -> dict[str, object]:
    hashes = {
        name: sha256_frozen_result_json(project_root / "results" / f"{name}.json")
        for name in (
            "citation_audit_cross_validation",
            "final_evaluation_dev",
            "controlled_experiments_dev",
        )
    }
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "label": GATE_LABEL,
        "frozen_results": {
            "citation_audit_cross_validation": {
                "path": "../results/citation_audit_cross_validation.json",
                "sha256": hashes["citation_audit_cross_validation"],
            },
            "final_evaluation_dev": {
                "path": "../results/final_evaluation_dev.json",
                "sha256": hashes["final_evaluation_dev"],
            },
            "controlled_experiments_dev": {
                "path": "../results/controlled_experiments_dev.json",
                "sha256": hashes["controlled_experiments_dev"],
            },
        },
        "held_out_development": [
            {
                "result": "final_evaluation_dev",
                "label_field": "evaluation_label",
                "expected_label": "held_out_development_evaluation",
                "independent_test_field": "is_independent_test",
            },
            {
                "result": "controlled_experiments_dev",
                "label_field": "evaluation_label",
                "expected_label": "held_out_development_evaluation",
                "independent_test_field": "is_independent_test",
            },
        ],
        "adversarial_suite": {
            "result": "controlled_experiments_dev",
            "field_path": ["adversarial_evaluator_suite", "all_passed"],
        },
        "report_consistency": [
            {
                "result": "final_evaluation_dev",
                "report_path": "../reports/phase_07_final_evaluation.md",
                "agent_trace_path": "../agent_trace/phase_07_final_evaluation.md",
            },
            {
                "result": "controlled_experiments_dev",
                "report_path": "../reports/phase_08_controlled_experiments.md",
                "agent_trace_path": "../agent_trace/phase_08_controlled_experiments.md",
            },
        ],
        "provenance_scan_results": [
            "citation_audit_cross_validation",
            "final_evaluation_dev",
            "controlled_experiments_dev",
        ],
    }


def _build_fixture(tmp_path: Path) -> Path:
    """Build a small, self-consistent Phase 6-8 evidence fixture and its config.

    ``citation_audit_cross_validation`` deliberately reuses a relative,
    backslash-separated Windows-style path (as the real frozen Phase 6
    artifact does) to prove the gate accepts it while still rejecting a truly
    absolute or drive-letter path.
    """
    _write_json(
        tmp_path / "results" / "citation_audit_cross_validation.json",
        {
            "schema_version": "evidence_agent_citation_audit_calibration_v1",
            "data": {
                "main_training_claims": {
                    "path": r"data\raw\scifact\data\claims_train.jsonl",
                }
            },
        },
    )
    _write_json(
        tmp_path / "results" / "final_evaluation_dev.json",
        {
            "evaluation_label": "held_out_development_evaluation",
            "is_independent_test": False,
            "output": {"report_path": "reports/phase_07_final_evaluation.md"},
        },
    )
    _write_json(
        tmp_path / "results" / "controlled_experiments_dev.json",
        {
            "evaluation_label": "held_out_development_evaluation",
            "is_independent_test": False,
            "adversarial_evaluator_suite": {"all_passed": True},
            "output": {"report_path": "reports/phase_08_controlled_experiments.md"},
        },
    )
    final_sha256 = sha256_frozen_result_json(tmp_path / "results" / "final_evaluation_dev.json")
    controlled_sha256 = sha256_frozen_result_json(tmp_path / "results" / "controlled_experiments_dev.json")
    _write_text(
        tmp_path / "reports" / "phase_07_final_evaluation.md",
        f"# Phase 07\n\nResult JSON SHA-256: `{final_sha256}`\n",
    )
    _write_text(
        tmp_path / "agent_trace" / "phase_07_final_evaluation.md",
        f"# Phase 07 trace\n\nResult JSON SHA-256: `{final_sha256}`\n",
    )
    _write_text(
        tmp_path / "reports" / "phase_08_controlled_experiments.md",
        f"# Phase 08\n\nResult JSON SHA-256: `{controlled_sha256}`\n",
    )
    _write_text(
        tmp_path / "agent_trace" / "phase_08_controlled_experiments.md",
        f"# Phase 08 trace\n\nResult JSON SHA-256: `{controlled_sha256}`\n",
    )
    config_path = tmp_path / "configs" / "reproducibility.yaml"
    _write_yaml(config_path, _config_payload(tmp_path))
    return config_path


def test_load_config_resolves_paths_relative_to_config_file(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)

    config = load_reproducibility_config(config_path)

    assert config.frozen_results["final_evaluation_dev"].path == (
        tmp_path / "results" / "final_evaluation_dev.json"
    ).resolve()
    assert config.report_consistency[0].report_path == (
        tmp_path / "reports" / "phase_07_final_evaluation.md"
    ).resolve()
    assert config.project_root == tmp_path.resolve()


def test_load_config_rejects_unknown_schema_version(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "not-a-schema"
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="schema_version"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_wrong_label(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["label"] = "something_else"
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="label"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["unexpected_field"] = True
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="unknown"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_reference_to_undeclared_frozen_result(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["adversarial_suite"]["result"] = "not_declared"
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="not_declared"):
        load_reproducibility_config(config_path)


@pytest.mark.parametrize(
    "result_name",
    ["citation_audit_cross_validation", "final_evaluation_dev", "controlled_experiments_dev"],
)
def test_gate_rejects_each_frozen_hash_mismatch(tmp_path: Path, result_name: str) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["frozen_results"][result_name]["sha256"] = "0" * 64
    _write_yaml(config_path, payload)
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match=result_name):
        run_reproducibility_check(config)


def test_sha256_frozen_result_json_is_identical_for_lf_and_crlf_checkouts(tmp_path: Path) -> None:
    """Cross-platform regression: CRLF (Windows) and LF (Linux CI) checkouts of the
    same committed frozen result JSON must hash to the same digest."""
    content = '{\n  "a": 1,\n  "b": [2, 3]\n}\n'
    lf_path = tmp_path / "lf.json"
    crlf_path = tmp_path / "crlf.json"
    lf_path.write_bytes(content.encode("utf-8"))
    crlf_path.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))

    assert sha256_frozen_result_json(lf_path) == sha256_frozen_result_json(crlf_path)


def test_sha256_frozen_result_json_still_detects_real_content_changes(tmp_path: Path) -> None:
    """A genuine content change -- not just a line-ending difference -- must still
    change the digest, so the normalization cannot mask a real evidence edit."""
    original_path = tmp_path / "original.json"
    changed_path = tmp_path / "changed.json"
    original_path.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    changed_path.write_bytes(b'{\r\n  "a": 2\r\n}\r\n')

    assert sha256_frozen_result_json(original_path) != sha256_frozen_result_json(changed_path)


def test_gate_rejects_non_held_out_development_label(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "final_evaluation_dev.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["evaluation_label"] = "independent_test"
    _write_json(result_path, payload)
    _write_yaml(config_path, _config_payload(tmp_path))
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="held_out_development_evaluation"):
        run_reproducibility_check(config)


def test_gate_rejects_is_independent_test_true(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "controlled_experiments_dev.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["is_independent_test"] = True
    _write_json(result_path, payload)
    _write_yaml(config_path, _config_payload(tmp_path))
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="is_independent_test"):
        run_reproducibility_check(config)


def test_gate_rejects_failed_adversarial_suite(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "controlled_experiments_dev.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["adversarial_evaluator_suite"]["all_passed"] = False
    _write_json(result_path, payload)
    _write_yaml(config_path, _config_payload(tmp_path))
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="adversarial"):
        run_reproducibility_check(config)


def test_gate_rejects_report_missing_result_hash(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    report_path = tmp_path / "reports" / "phase_07_final_evaluation.md"
    _write_text(report_path, "# Phase 07\n\nNo hash here.\n")

    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="does not contain"):
        run_reproducibility_check(config)


def test_gate_rejects_agent_trace_missing_result_hash(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    trace_path = tmp_path / "agent_trace" / "phase_08_controlled_experiments.md"
    _write_text(trace_path, "# Phase 08 trace\n\nNo hash here.\n")

    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="does not contain"):
        run_reproducibility_check(config)


@pytest.mark.parametrize(
    "accepted_path",
    [
        "artifacts/folds/file.json",
        r"artifacts\folds\file.json",
        r"data\raw\scifact\data\claims_train.jsonl",
    ],
)
def test_is_forbidden_committed_path_accepts_relative_paths(accepted_path: str) -> None:
    assert _is_forbidden_committed_path(accepted_path) is False


def test_is_forbidden_committed_path_accepts_a_relative_backslash_path() -> None:
    """Direct regression: a relative Windows-style backslash path is accepted.

    This is exactly the shape the real, frozen, immutable Phase 6 fold
    artifacts use (they were written on Windows), so the gate must not
    reject them.
    """
    assert _is_forbidden_committed_path(r"artifacts\citation_audit_cv\fold_1_verifier.joblib") is False


@pytest.mark.parametrize(
    ("rejected_path", "reason"),
    [
        ("/tmp/file.json", "POSIX-absolute path"),
        ("/home/someone/project/data/raw/scifact/data/claims_train.jsonl", "POSIX-absolute path"),
        (r"C:\Users\someone\project\data\raw\scifact\data\claims_train.jsonl", "drive-letter path"),
        ("C:/Users/someone/project/data/raw/scifact/data/claims_train.jsonl", "drive-letter path"),
        ("C:relative", "drive-letter path without a root"),
        (r"\rooted\file.json", "Windows-rooted path with no drive"),
        (r"\\server\share\file.json", "UNC path"),
        ("", "empty path"),
    ],
)
def test_is_forbidden_committed_path_rejects_unsafe_paths(rejected_path: str, reason: str) -> None:
    assert _is_forbidden_committed_path(rejected_path) is True, reason


@pytest.mark.parametrize(
    "forbidden_path",
    [
        r"C:\Users\someone\project\data\raw\scifact\data\claims_train.jsonl",
        "C:/Users/someone/project/data/raw/scifact/data/claims_train.jsonl",
        "C:relative",
        "/home/someone/project/data/raw/scifact/data/claims_train.jsonl",
        r"\rooted\file.json",
        r"\\server\share\file.json",
    ],
)
def test_gate_rejects_absolute_or_drive_letter_provenance_path(
    tmp_path: Path, forbidden_path: str
) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "citation_audit_cross_validation.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["data"]["main_training_claims"]["path"] = forbidden_path
    _write_json(result_path, payload)
    _write_yaml(config_path, _config_payload(tmp_path))
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="portable"):
        run_reproducibility_check(config)


def test_gate_accepts_relative_backslash_provenance_path(tmp_path: Path) -> None:
    """Regression: the real frozen Phase 6 artifact uses relative backslash paths."""
    config_path = _build_fixture(tmp_path)
    config = load_reproducibility_config(config_path)

    manifest = run_reproducibility_check(config)

    assert manifest["ok"] is True


def test_gate_manifest_reports_every_check(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    config = load_reproducibility_config(config_path)

    manifest = run_reproducibility_check(config)

    assert manifest["ok"] is True
    assert len(manifest["held_out_development_checks"]) == 2
    assert all(check["passed"] for check in manifest["held_out_development_checks"])
    assert manifest["adversarial_suite_check"]["passed"] is True
    assert len(manifest["report_consistency_checks"]) == 2


def test_run_reproducibility_command_writes_only_to_output_dir(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    output_dir = tmp_path / "artifacts" / "phase09_check"
    before = {
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and output_dir not in path.parents
    }

    manifest = run_reproducibility_command(config_path, output_dir)

    after = {
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and output_dir not in path.parents
    }
    assert before == after
    manifest_path = output_dir / MANIFEST_FILENAME
    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["ok"] is True
    assert manifest["ok"] is True


def test_cli_reproducibility_runs_end_to_end_on_temporary_fixture(
    tmp_path: Path, capsys
) -> None:
    config_path = _build_fixture(tmp_path)
    output_dir = tmp_path / "artifacts" / "phase09_cli_check"
    capsys.readouterr()

    exit_code = main(
        ["reproducibility", "--config", str(config_path), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert Path(printed["manifest_path"]).is_file()


def test_cli_reproducibility_requires_output_dir(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)

    with pytest.raises(SystemExit):
        main(["reproducibility", "--config", str(config_path)])


def test_cli_reproducibility_propagates_gate_failure(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "controlled_experiments_dev.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["adversarial_evaluator_suite"]["all_passed"] = False
    _write_json(result_path, payload)
    _write_yaml(config_path, _config_payload(tmp_path))
    output_dir = tmp_path / "artifacts" / "phase09_cli_failure"

    with pytest.raises(ReproducibilityError):
        main(["reproducibility", "--config", str(config_path), "--output-dir", str(output_dir)])
    assert not output_dir.exists()


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReproducibilityConfigError, match="Unable to read"):
        load_reproducibility_config(tmp_path / "does_not_exist.yaml")


def test_load_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "reproducibility.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("frozen_results: [unterminated\n", encoding="utf-8")

    with pytest.raises(ReproducibilityConfigError, match="Invalid YAML"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_non_mapping_document(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "reproducibility.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ReproducibilityConfigError, match="must be a mapping"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_empty_frozen_results(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["frozen_results"] = {}
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="at least one entry"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_bad_sha256_length(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["frozen_results"]["final_evaluation_dev"]["sha256"] = "not-a-digest"
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="64-character hex digest"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_non_hexadecimal_sha256(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["frozen_results"]["final_evaluation_dev"]["sha256"] = "z" * 64
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="hexadecimal"):
        load_reproducibility_config(config_path)


@pytest.mark.parametrize("bad_value", [None, "not-a-list", 42])
def test_load_config_rejects_non_list_held_out_development(tmp_path: Path, bad_value: object) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["held_out_development"] = bad_value
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="held_out_development must be a list"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_non_list_report_consistency(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["report_consistency"] = "not-a-list"
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="report_consistency must be a list"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_invalid_provenance_scan_results(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["provenance_scan_results"] = ["", "final_evaluation_dev"]
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="provenance_scan_results"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_incomplete_held_out_development_entry(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["held_out_development"][0]["expected_label"] = ""
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="non-empty strings"):
        load_reproducibility_config(config_path)


def test_load_config_rejects_empty_adversarial_field_path(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["adversarial_suite"]["field_path"] = []
    _write_yaml(config_path, payload)

    with pytest.raises(ReproducibilityConfigError, match="field_path"):
        load_reproducibility_config(config_path)


def test_gate_rejects_missing_frozen_result_file(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    config = load_reproducibility_config(config_path)
    (tmp_path / "results" / "final_evaluation_dev.json").unlink()

    with pytest.raises(ReproducibilityError, match="unable to read"):
        run_reproducibility_check(config)


def test_gate_rejects_malformed_frozen_result_json(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "final_evaluation_dev.json"
    result_path.write_text("{not valid json", encoding="utf-8")
    _write_yaml(config_path, _config_payload(tmp_path))
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="unable to parse"):
        run_reproducibility_check(config)


def test_gate_rejects_missing_adversarial_field(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    result_path = tmp_path / "results" / "controlled_experiments_dev.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    del payload["adversarial_evaluator_suite"]["all_passed"]
    _write_json(result_path, payload)
    _write_yaml(config_path, _config_payload(tmp_path))
    config = load_reproducibility_config(config_path)

    with pytest.raises(ReproducibilityError, match="missing adversarial suite field"):
        run_reproducibility_check(config)


def test_gate_rejects_missing_report_file(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    config = load_reproducibility_config(config_path)
    (tmp_path / "reports" / "phase_07_final_evaluation.md").unlink()

    with pytest.raises(ReproducibilityError, match="unable to read report"):
        run_reproducibility_check(config)


def test_gate_rejects_missing_agent_trace_file(tmp_path: Path) -> None:
    config_path = _build_fixture(tmp_path)
    config = load_reproducibility_config(config_path)
    (tmp_path / "agent_trace" / "phase_07_final_evaluation.md").unlink()

    with pytest.raises(ReproducibilityError, match="unable to read agent trace"):
        run_reproducibility_check(config)


def test_real_committed_phase_06_08_evidence_passes_the_gate() -> None:
    """The actual committed configs/reproducibility.yaml must pass against real evidence."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_reproducibility_config(repo_root / "configs" / "reproducibility.yaml")

    manifest = run_reproducibility_check(config)

    assert manifest["ok"] is True
