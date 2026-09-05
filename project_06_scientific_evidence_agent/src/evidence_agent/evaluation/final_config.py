"""Safe, schema-validated configuration for the Phase 7 final evaluation.

Every path in ``configs/final.yaml`` is resolved relative to the directory
containing that config file, never the process working directory. This module
only parses and validates the *shape* of the configuration; it does not read
or hash the referenced artifacts. Hash validation happens in
:mod:`evidence_agent.evaluation.final`, immediately before any artifact loads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


CONFIG_SCHEMA_VERSION = "evidence_agent_final_evaluation_config_v1"
EVALUATION_LABEL = "held_out_development_evaluation"

_REQUIRED_ARTIFACTS = (
    "corpus",
    "claims_dev",
    "bm25_index",
    "verifier_model",
    "calibration_report",
)
_OPTIONAL_ARTIFACTS = ("train_claims",)

# Phase 6 result files that Phase 7 must never overwrite. Resolved relative to
# the config file's grandparent directory, matching this project's fixed
# layout where ``configs/`` and ``results/`` are sibling directories.
_FORBIDDEN_RESULT_RELATIVE_PATHS = (
    Path("results/citation_audit_dev.json"),
    Path("results/citation_audit_cross_validation.json"),
)


class FinalEvaluationConfigError(ValueError):
    """Raised when ``configs/final.yaml`` is missing, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A declared input path paired with its expected SHA-256 digest."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        digest = self.sha256
        if not isinstance(digest, str) or len(digest) != 64:
            raise FinalEvaluationConfigError(
                f"Artifact SHA-256 for {self.path} must be a 64-character hex digest."
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise FinalEvaluationConfigError(
                f"Artifact SHA-256 for {self.path} must be hexadecimal."
            ) from error
        object.__setattr__(self, "sha256", digest.lower())


@dataclass(frozen=True, slots=True)
class FinalEvaluationArtifacts:
    """Every prebuilt, fixed artifact the final evaluation is allowed to load."""

    corpus: ArtifactReference
    claims_dev: ArtifactReference
    bm25_index: ArtifactReference
    verifier_model: ArtifactReference
    calibration_report: ArtifactReference
    train_claims: ArtifactReference | None = None


@dataclass(frozen=True, slots=True)
class FinalEvaluationRuntime:
    """Fixed runtime settings applied to the frozen verifier bundle."""

    retrieval_k: int

    def __post_init__(self) -> None:
        if isinstance(self.retrieval_k, bool) or not isinstance(self.retrieval_k, int) or self.retrieval_k <= 0:
            raise FinalEvaluationConfigError("runtime.retrieval_k must be a positive integer.")


@dataclass(frozen=True, slots=True)
class FinalEvaluationBootstrap:
    """Deterministic paired-bootstrap confidence-interval settings."""

    enabled: bool
    resamples: int
    seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise FinalEvaluationConfigError("bootstrap.enabled must be a boolean.")
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int) or self.resamples <= 0:
            raise FinalEvaluationConfigError("bootstrap.resamples must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise FinalEvaluationConfigError("bootstrap.seed must be a non-negative integer.")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise FinalEvaluationConfigError("bootstrap.confidence_level must lie in (0, 1).")


@dataclass(frozen=True, slots=True)
class FinalEvaluationOutput:
    """Where the Phase 7 result JSON, raw trace, and narrative docs will live."""

    result_path: Path
    trace_path: Path
    report_path: Path
    agent_trace_path: Path

    def __post_init__(self) -> None:
        paths = {
            "result_path": self.result_path,
            "trace_path": self.trace_path,
            "report_path": self.report_path,
            "agent_trace_path": self.agent_trace_path,
        }
        if len({str(path) for path in paths.values()}) != len(paths):
            raise FinalEvaluationConfigError("output paths must all be distinct.")


@dataclass(frozen=True, slots=True)
class FinalEvaluationConfig:
    """A fully validated, path-resolved Phase 7 final-evaluation configuration."""

    config_path: Path
    project_root: Path
    artifacts: FinalEvaluationArtifacts
    runtime: FinalEvaluationRuntime
    bootstrap: FinalEvaluationBootstrap
    output: FinalEvaluationOutput


def _resolve_path(config_dir: Path, raw_value: object, field_name: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise FinalEvaluationConfigError(f"{field_name} must be a non-empty string path.")
    candidate = Path(raw_value)
    resolved = candidate if candidate.is_absolute() else (config_dir / candidate)
    return resolved.resolve()


def _require_mapping(payload: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise FinalEvaluationConfigError(f"{field_name} must be a mapping.")
    return payload


def _parse_artifact_reference(
    config_dir: Path, section: object, field_name: str
) -> ArtifactReference:
    mapping = _require_mapping(section, field_name)
    path = _resolve_path(config_dir, mapping.get("path"), f"{field_name}.path")
    sha256 = mapping.get("sha256")
    if not isinstance(sha256, str):
        raise FinalEvaluationConfigError(f"{field_name}.sha256 must be a string.")
    return ArtifactReference(path=path, sha256=sha256)


def load_final_evaluation_config(config_path: Path) -> FinalEvaluationConfig:
    """Safely load, schema-validate, and path-resolve ``configs/final.yaml``."""
    config_path = Path(config_path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FinalEvaluationConfigError(
            f"Unable to read final-evaluation config {config_path}: {error}"
        ) from error
    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise FinalEvaluationConfigError(f"Invalid YAML in {config_path}: {error}") from error

    payload = _require_mapping(payload, str(config_path))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise FinalEvaluationConfigError(
            f"{config_path} must declare schema_version '{CONFIG_SCHEMA_VERSION}'."
        )
    if payload.get("label") != EVALUATION_LABEL:
        raise FinalEvaluationConfigError(
            f"{config_path} must declare label '{EVALUATION_LABEL}'."
        )

    config_dir = config_path.resolve().parent

    artifacts_section = _require_mapping(payload.get("artifacts"), "artifacts")
    missing_required = [name for name in _REQUIRED_ARTIFACTS if name not in artifacts_section]
    if missing_required:
        raise FinalEvaluationConfigError(
            f"artifacts is missing required entries: {sorted(missing_required)}"
        )
    unknown = set(artifacts_section) - set(_REQUIRED_ARTIFACTS) - set(_OPTIONAL_ARTIFACTS)
    if unknown:
        raise FinalEvaluationConfigError(f"artifacts has unknown entries: {sorted(unknown)}")
    artifacts = FinalEvaluationArtifacts(
        corpus=_parse_artifact_reference(config_dir, artifacts_section["corpus"], "artifacts.corpus"),
        claims_dev=_parse_artifact_reference(
            config_dir, artifacts_section["claims_dev"], "artifacts.claims_dev"
        ),
        bm25_index=_parse_artifact_reference(
            config_dir, artifacts_section["bm25_index"], "artifacts.bm25_index"
        ),
        verifier_model=_parse_artifact_reference(
            config_dir, artifacts_section["verifier_model"], "artifacts.verifier_model"
        ),
        calibration_report=_parse_artifact_reference(
            config_dir, artifacts_section["calibration_report"], "artifacts.calibration_report"
        ),
        train_claims=(
            _parse_artifact_reference(
                config_dir, artifacts_section["train_claims"], "artifacts.train_claims"
            )
            if "train_claims" in artifacts_section
            else None
        ),
    )

    runtime_section = _require_mapping(payload.get("runtime"), "runtime")
    runtime = FinalEvaluationRuntime(retrieval_k=runtime_section.get("retrieval_k"))

    bootstrap_section = _require_mapping(payload.get("bootstrap"), "bootstrap")
    bootstrap = FinalEvaluationBootstrap(
        enabled=bootstrap_section.get("enabled"),
        resamples=bootstrap_section.get("resamples"),
        seed=bootstrap_section.get("seed"),
        confidence_level=bootstrap_section.get("confidence_level"),
    )

    output_section = _require_mapping(payload.get("output"), "output")
    output = FinalEvaluationOutput(
        result_path=_resolve_path(config_dir, output_section.get("result_path"), "output.result_path"),
        trace_path=_resolve_path(config_dir, output_section.get("trace_path"), "output.trace_path"),
        report_path=_resolve_path(config_dir, output_section.get("report_path"), "output.report_path"),
        agent_trace_path=_resolve_path(
            config_dir, output_section.get("agent_trace_path"), "output.agent_trace_path"
        ),
    )

    project_root = config_dir.parent
    forbidden_paths = {
        (project_root / relative_path).resolve() for relative_path in _FORBIDDEN_RESULT_RELATIVE_PATHS
    }
    for field_name, path in (
        ("output.result_path", output.result_path),
        ("output.trace_path", output.trace_path),
        ("output.report_path", output.report_path),
        ("output.agent_trace_path", output.agent_trace_path),
    ):
        if path in forbidden_paths:
            raise FinalEvaluationConfigError(
                f"{field_name} must not reuse a Phase 6 result path: {path}"
            )

    return FinalEvaluationConfig(
        config_path=config_path,
        project_root=project_root,
        artifacts=artifacts,
        runtime=runtime,
        bootstrap=bootstrap,
        output=output,
    )
