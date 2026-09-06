"""Phase 9 clean-environment reproducibility gate.

This module verifies -- without downloading raw data, retraining any model,
recalibrating a policy, or overwriting a prior result -- that the frozen
Phase 6-8 evidence committed to the repository is internally consistent:

* each frozen result JSON matches its declared SHA-256 digest;
* Phase 7 and Phase 8 are labelled a held-out *development* evaluation, never
  an independent test;
* Phase 8's adversarial evaluator-regression suite passed;
* the committed Markdown report and agent trace for each labelled result
  contain that same result's SHA-256, so the narrative and the machine record
  cannot silently drift apart;
* every path recorded inside a frozen result is portable and project-relative
  -- a relative POSIX path or a relative Windows-style backslash path is
  accepted, but a POSIX-absolute path, a drive-letter path, a Windows-rooted
  path, or a UNC path is rejected, since any of those would leak a
  contributor's local machine layout into committed evidence.

All configuration paths are resolved relative to the configuration file
itself, never the process's current working directory, so the gate behaves
identically regardless of where it is invoked from. The check only reads
committed, Git-tracked files (``results/*.json``, ``reports/*.md``,
``agent_trace/*.md``); it never requires the ignored raw SciFact release,
trained model artifacts, or runtime traces.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from evidence_agent.data.acquisition import sha256_file
from evidence_agent.evaluation.verification import write_verification_report as write_json_report


CONFIG_SCHEMA_VERSION = "evidence_agent_reproducibility_config_v1"
MANIFEST_SCHEMA_VERSION = "evidence_agent_reproducibility_manifest_v1"
GATE_LABEL = "clean_environment_reproducibility_gate"
MANIFEST_FILENAME = "reproducibility_manifest.json"

_DRIVE_LETTER_PATTERN = re.compile(r"^[A-Za-z]:")


class ReproducibilityConfigError(ValueError):
    """Raised when ``configs/reproducibility.yaml`` is malformed or unsafe."""


class ReproducibilityError(ValueError):
    """Raised when a committed Phase 6-8 evidence artifact fails the gate."""


def _require_mapping(payload: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ReproducibilityConfigError(f"{field_name} must be a mapping.")
    return payload


def _require_exact_keys(mapping: Mapping[str, object], expected: set[str], field_name: str) -> None:
    unknown = set(mapping) - expected
    missing = expected - set(mapping)
    if unknown or missing:
        problems: list[str] = []
        if missing:
            problems.append(f"missing {sorted(missing)}")
        if unknown:
            problems.append(f"unknown {sorted(unknown)}")
        raise ReproducibilityConfigError(f"{field_name} has {'; '.join(problems)} fields.")


def _resolve_path(config_dir: Path, raw_value: object, field_name: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ReproducibilityConfigError(f"{field_name} must be a non-empty string path.")
    candidate = Path(raw_value)
    resolved = candidate if candidate.is_absolute() else config_dir / candidate
    return resolved.resolve()


def _relative_posix_path(path: Path, project_root: Path) -> str:
    resolved = path if path.is_absolute() else path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass(frozen=True, slots=True)
class FrozenResultRef:
    """One committed JSON result and the SHA-256 digest it must match."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        digest = self.sha256
        if not isinstance(digest, str) or len(digest) != 64:
            raise ReproducibilityConfigError(
                f"Frozen result SHA-256 for {self.path} must be a 64-character hex digest."
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise ReproducibilityConfigError(
                f"Frozen result SHA-256 for {self.path} must be hexadecimal."
            ) from error
        object.__setattr__(self, "sha256", digest.lower())


@dataclass(frozen=True, slots=True)
class HeldOutDevelopmentCheck:
    """A held-out-development label assertion on one frozen result."""

    result: str
    label_field: str
    expected_label: str
    independent_test_field: str


@dataclass(frozen=True, slots=True)
class AdversarialSuiteCheck:
    """A required ``all_passed is True`` assertion at a nested field path."""

    result: str
    field_path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReportConsistencyCheck:
    """A frozen result's SHA-256 must appear in its report and agent trace."""

    result: str
    report_path: Path
    agent_trace_path: Path


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Fully path-resolved and schema-validated Phase 9 configuration."""

    config_path: Path
    project_root: Path
    frozen_results: Mapping[str, FrozenResultRef]
    held_out_development: tuple[HeldOutDevelopmentCheck, ...]
    adversarial_suite: AdversarialSuiteCheck
    report_consistency: tuple[ReportConsistencyCheck, ...]
    provenance_scan_results: tuple[str, ...]


def _parse_frozen_result(config_dir: Path, section: object, field_name: str) -> FrozenResultRef:
    mapping = _require_mapping(section, field_name)
    _require_exact_keys(mapping, {"path", "sha256"}, field_name)
    sha256 = mapping.get("sha256")
    if not isinstance(sha256, str):
        raise ReproducibilityConfigError(f"{field_name}.sha256 must be a string.")
    return FrozenResultRef(
        path=_resolve_path(config_dir, mapping.get("path"), f"{field_name}.path"),
        sha256=sha256,
    )


def _parse_held_out_development(section: object, index: int) -> HeldOutDevelopmentCheck:
    mapping = _require_mapping(section, f"held_out_development[{index}]")
    _require_exact_keys(
        mapping,
        {"result", "label_field", "expected_label", "independent_test_field"},
        f"held_out_development[{index}]",
    )
    result = mapping.get("result")
    label_field = mapping.get("label_field")
    expected_label = mapping.get("expected_label")
    independent_test_field = mapping.get("independent_test_field")
    if not all(isinstance(value, str) and value for value in (result, label_field, expected_label, independent_test_field)):
        raise ReproducibilityConfigError(
            f"held_out_development[{index}] fields must all be non-empty strings."
        )
    return HeldOutDevelopmentCheck(
        result=result,
        label_field=label_field,
        expected_label=expected_label,
        independent_test_field=independent_test_field,
    )


def _parse_adversarial_suite(section: object) -> AdversarialSuiteCheck:
    mapping = _require_mapping(section, "adversarial_suite")
    _require_exact_keys(mapping, {"result", "field_path"}, "adversarial_suite")
    result = mapping.get("result")
    field_path = mapping.get("field_path")
    if not isinstance(result, str) or not result:
        raise ReproducibilityConfigError("adversarial_suite.result must be a non-empty string.")
    if (
        not isinstance(field_path, Sequence)
        or isinstance(field_path, (str, bytes))
        or not field_path
        or not all(isinstance(item, str) and item for item in field_path)
    ):
        raise ReproducibilityConfigError(
            "adversarial_suite.field_path must be a non-empty list of non-empty strings."
        )
    return AdversarialSuiteCheck(result=result, field_path=tuple(field_path))


def _parse_report_consistency(config_dir: Path, section: object, index: int) -> ReportConsistencyCheck:
    mapping = _require_mapping(section, f"report_consistency[{index}]")
    _require_exact_keys(mapping, {"result", "report_path", "agent_trace_path"}, f"report_consistency[{index}]")
    result = mapping.get("result")
    if not isinstance(result, str) or not result:
        raise ReproducibilityConfigError(f"report_consistency[{index}].result must be a non-empty string.")
    return ReportConsistencyCheck(
        result=result,
        report_path=_resolve_path(config_dir, mapping.get("report_path"), f"report_consistency[{index}].report_path"),
        agent_trace_path=_resolve_path(
            config_dir, mapping.get("agent_trace_path"), f"report_consistency[{index}].agent_trace_path"
        ),
    )


def load_reproducibility_config(config_path: Path) -> ReproducibilityConfig:
    """Load and schema-validate the Phase 9 configuration without reading evidence."""
    config_path = Path(config_path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReproducibilityConfigError(
            f"Unable to read reproducibility config {config_path}: {error}"
        ) from error
    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ReproducibilityConfigError(f"Invalid YAML in {config_path}: {error}") from error

    payload = _require_mapping(payload, str(config_path))
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "label",
            "frozen_results",
            "held_out_development",
            "adversarial_suite",
            "report_consistency",
            "provenance_scan_results",
        },
        str(config_path),
    )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ReproducibilityConfigError(
            f"{config_path} must declare schema_version '{CONFIG_SCHEMA_VERSION}'."
        )
    if payload.get("label") != GATE_LABEL:
        raise ReproducibilityConfigError(f"{config_path} must declare label '{GATE_LABEL}'.")

    config_dir = config_path.resolve().parent
    project_root = config_dir.parent

    frozen_section = _require_mapping(payload.get("frozen_results"), "frozen_results")
    if not frozen_section:
        raise ReproducibilityConfigError("frozen_results must declare at least one entry.")
    frozen_results = {
        name: _parse_frozen_result(config_dir, section, f"frozen_results.{name}")
        for name, section in frozen_section.items()
    }

    held_out_section = payload.get("held_out_development")
    if not isinstance(held_out_section, Sequence) or isinstance(held_out_section, (str, bytes)):
        raise ReproducibilityConfigError("held_out_development must be a list.")
    held_out_development = tuple(
        _parse_held_out_development(entry, index) for index, entry in enumerate(held_out_section)
    )

    adversarial_suite = _parse_adversarial_suite(payload.get("adversarial_suite"))

    report_section = payload.get("report_consistency")
    if not isinstance(report_section, Sequence) or isinstance(report_section, (str, bytes)):
        raise ReproducibilityConfigError("report_consistency must be a list.")
    report_consistency = tuple(
        _parse_report_consistency(config_dir, entry, index) for index, entry in enumerate(report_section)
    )

    provenance_section = payload.get("provenance_scan_results")
    if (
        not isinstance(provenance_section, Sequence)
        or isinstance(provenance_section, (str, bytes))
        or not all(isinstance(item, str) and item for item in provenance_section)
    ):
        raise ReproducibilityConfigError("provenance_scan_results must be a list of non-empty strings.")
    provenance_scan_results = tuple(provenance_section)

    for name in (*(check.result for check in held_out_development), adversarial_suite.result,
                 *(check.result for check in report_consistency), *provenance_scan_results):
        if name not in frozen_results:
            raise ReproducibilityConfigError(f"'{name}' is not declared under frozen_results.")

    return ReproducibilityConfig(
        config_path=config_path.resolve(),
        project_root=project_root,
        frozen_results=frozen_results,
        held_out_development=held_out_development,
        adversarial_suite=adversarial_suite,
        report_consistency=report_consistency,
        provenance_scan_results=provenance_scan_results,
    )


def _validate_frozen_result_hashes(config: ReproducibilityConfig) -> dict[str, Mapping[str, object]]:
    """Validate every declared digest, then load and return each result's JSON."""
    loaded: dict[str, Mapping[str, object]] = {}
    for name, ref in config.frozen_results.items():
        try:
            actual_sha256 = sha256_file(ref.path)
        except OSError as error:
            raise ReproducibilityError(
                f"{name}: unable to read frozen result at {ref.path}: {error}"
            ) from error
        if actual_sha256 != ref.sha256:
            raise ReproducibilityError(
                f"{name}: SHA-256 mismatch for {ref.path} "
                f"(declared {ref.sha256}, actual {actual_sha256})."
            )
        try:
            loaded[name] = json.loads(ref.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReproducibilityError(f"{name}: unable to parse frozen result JSON: {error}") from error
    return loaded


def _validate_held_out_development(
    config: ReproducibilityConfig, loaded_results: Mapping[str, Mapping[str, object]]
) -> None:
    for check in config.held_out_development:
        payload = loaded_results[check.result]
        label_value = payload.get(check.label_field)
        if label_value != check.expected_label:
            raise ReproducibilityError(
                f"{check.result}.{check.label_field} must equal '{check.expected_label}' "
                f"(held-out development, not an independent test); got {label_value!r}."
            )
        independent_test_value = payload.get(check.independent_test_field)
        if independent_test_value is not False:
            raise ReproducibilityError(
                f"{check.result}.{check.independent_test_field} must be False; "
                f"got {independent_test_value!r}."
            )


def _validate_adversarial_suite(
    config: ReproducibilityConfig, loaded_results: Mapping[str, Mapping[str, object]]
) -> None:
    check = config.adversarial_suite
    value: object = loaded_results[check.result]
    for key in check.field_path:
        if not isinstance(value, Mapping) or key not in value:
            raise ReproducibilityError(
                f"{check.result} is missing adversarial suite field '{'.'.join(check.field_path)}'."
            )
        value = value[key]
    if value is not True:
        raise ReproducibilityError(
            f"{check.result}.{'.'.join(check.field_path)} must be True "
            f"(the adversarial evaluator suite must pass); got {value!r}."
        )


def _validate_report_consistency(
    config: ReproducibilityConfig, loaded_results: Mapping[str, Mapping[str, object]]
) -> None:
    for check in config.report_consistency:
        expected_sha256 = config.frozen_results[check.result].sha256
        try:
            report_text = check.report_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ReproducibilityError(
                f"{check.result}: unable to read report {check.report_path}: {error}"
            ) from error
        try:
            trace_text = check.agent_trace_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ReproducibilityError(
                f"{check.result}: unable to read agent trace {check.agent_trace_path}: {error}"
            ) from error
        if expected_sha256 not in report_text:
            raise ReproducibilityError(
                f"{check.report_path} does not contain the {check.result} result SHA-256 ({expected_sha256})."
            )
        if expected_sha256 not in trace_text:
            raise ReproducibilityError(
                f"{check.agent_trace_path} does not contain the {check.result} result SHA-256 ({expected_sha256})."
            )


def _is_forbidden_committed_path(value: str) -> bool:
    """Reject any path that would leak a contributor's local machine layout.

    Accepted: a relative POSIX path (``artifacts/folds/file.json``) and a
    relative Windows-style path using backslash separators
    (``artifacts\\folds\\file.json``), as recorded by some earlier, frozen
    Phase 6 fold artifacts written on Windows -- it carries no
    machine-specific root, so it is portable in the sense that matters here.

    Rejected: a POSIX-absolute path (``/tmp/file.json``), a drive-letter path
    whether rooted or not (``C:\\Users\\...``, ``C:relative``), a
    Windows-rooted path with no drive (``\\rooted\\file.json``), and a UNC
    path (``\\\\server\\share\\file.json``) -- the last two are both caught by
    the same "starts with a backslash" check, since a UNC path is simply a
    Windows-rooted path with two leading backslashes instead of one.
    """
    if not value:
        return True
    if value.startswith("/"):
        return True
    if value.startswith("\\"):
        return True
    if _DRIVE_LETTER_PATTERN.match(value):
        return True
    return False


def _iter_path_like_fields(
    payload: object, prefix: tuple[str, ...] = ()
) -> Sequence[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_str = str(key)
            new_prefix = prefix + (key_str,)
            if isinstance(value, str) and (key_str == "path" or key_str.endswith("_path")):
                found.append((new_prefix, value))
            else:
                found.extend(_iter_path_like_fields(value, new_prefix))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_iter_path_like_fields(value, prefix + (str(index),)))
    return found


def _validate_provenance_paths(
    config: ReproducibilityConfig, loaded_results: Mapping[str, Mapping[str, object]]
) -> None:
    for name in config.provenance_scan_results:
        payload = loaded_results[name]
        for field_path, value in _iter_path_like_fields(payload):
            if _is_forbidden_committed_path(value):
                raise ReproducibilityError(
                    f"{name}.{'.'.join(field_path)} is not a portable, project-relative "
                    f"POSIX path: {value!r}."
                )


def run_reproducibility_check(config: ReproducibilityConfig) -> dict[str, object]:
    """Run every Phase 9 gate check and return a manifest describing the result.

    Raises :class:`ReproducibilityError` on the first failing check. No
    ignored raw data, trained model, or runtime-trace artifact is required;
    only committed ``results/``, ``reports/``, and ``agent_trace/`` files are
    read.
    """
    loaded_results = _validate_frozen_result_hashes(config)
    _validate_held_out_development(config, loaded_results)
    _validate_adversarial_suite(config, loaded_results)
    _validate_report_consistency(config, loaded_results)
    _validate_provenance_paths(config, loaded_results)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "label": GATE_LABEL,
        "config_path": _relative_posix_path(config.config_path, config.project_root),
        "frozen_results": {
            name: {
                "path": _relative_posix_path(ref.path, config.project_root),
                "sha256": ref.sha256,
            }
            for name, ref in sorted(config.frozen_results.items())
        },
        "held_out_development_checks": [
            {
                "result": check.result,
                "label_field": check.label_field,
                "expected_label": check.expected_label,
                "independent_test_field": check.independent_test_field,
                "passed": True,
            }
            for check in config.held_out_development
        ],
        "adversarial_suite_check": {
            "result": config.adversarial_suite.result,
            "field_path": list(config.adversarial_suite.field_path),
            "passed": True,
        },
        "report_consistency_checks": [
            {
                "result": check.result,
                "report_path": _relative_posix_path(check.report_path, config.project_root),
                "agent_trace_path": _relative_posix_path(check.agent_trace_path, config.project_root),
                "passed": True,
            }
            for check in config.report_consistency
        ],
        "provenance_scan_results": list(config.provenance_scan_results),
        "ok": True,
    }


def run_reproducibility_command(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Thin CLI entry point: run the gate and write only to ``output_dir``."""
    config = load_reproducibility_config(config_path)
    manifest = run_reproducibility_check(config)
    output_dir = Path(output_dir)
    manifest_path = output_dir / MANIFEST_FILENAME
    write_json_report(manifest, manifest_path)
    return {**manifest, "manifest_path": str(manifest_path)}
