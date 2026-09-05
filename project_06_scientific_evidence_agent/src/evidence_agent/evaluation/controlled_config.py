"""Schema-validated configuration for Phase 8 controlled experiments.

Phase 8 compares a direct retrieval-to-citation baseline with the frozen,
cross-validated citation-audit policy.  The configuration is intentionally
separate from ``configs/final.yaml`` so a controlled experiment cannot replace
any Phase 6 or Phase 7 evidence artifact by accident.

All relative paths are resolved from the configuration file itself, never the
caller’s current working directory.  Hash validation is performed by
``evaluation.controlled`` immediately before an artifact is loaded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


CONFIG_SCHEMA_VERSION = "evidence_agent_controlled_experiments_config_v1"
EVALUATION_LABEL = "held_out_development_evaluation"

_REQUIRED_ARTIFACTS = (
    "corpus",
    "claims_dev",
    "bm25_index",
    "verifier_model",
    "calibration_report",
)
_OPTIONAL_ARTIFACTS = ("train_claims",)
_FORBIDDEN_OUTPUT_RELATIVE_PATHS = (
    Path("results/citation_audit_dev.json"),
    Path("results/citation_audit_cross_validation.json"),
    Path("results/final_evaluation_dev.json"),
    Path("artifacts/citation_audit_dev_trace.json"),
    Path("artifacts/final_evaluation_dev_trace.json"),
    Path("reports/phase_06_citation_audit.md"),
    Path("reports/phase_07_final_evaluation.md"),
    Path("agent_trace/phase_06_citation_audit.md"),
    Path("agent_trace/phase_07_final_evaluation.md"),
)


class ControlledExperimentsConfigError(ValueError):
    """Raised when a Phase 8 configuration is malformed or unsafe."""


def _require_exact_keys(
    mapping: Mapping[str, object], expected: set[str], field_name: str
) -> None:
    """Reject misspelled or silently ignored configuration fields."""
    unknown = set(mapping) - expected
    missing = expected - set(mapping)
    if unknown or missing:
        problems: list[str] = []
        if missing:
            problems.append(f"missing {sorted(missing)}")
        if unknown:
            problems.append(f"unknown {sorted(unknown)}")
        raise ControlledExperimentsConfigError(f"{field_name} has {'; '.join(problems)} fields.")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """One immutable input path and its expected SHA-256 digest."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        digest = self.sha256
        if not isinstance(digest, str) or len(digest) != 64:
            raise ControlledExperimentsConfigError(
                f"Artifact SHA-256 for {self.path} must be a 64-character hex digest."
            )
        try:
            int(digest, 16)
        except ValueError as error:
            raise ControlledExperimentsConfigError(
                f"Artifact SHA-256 for {self.path} must be hexadecimal."
            ) from error
        object.__setattr__(self, "sha256", digest.lower())


@dataclass(frozen=True, slots=True)
class ControlledExperimentArtifacts:
    """Every fixed artifact consumed by a controlled experiment."""

    corpus: ArtifactReference
    claims_dev: ArtifactReference
    bm25_index: ArtifactReference
    verifier_model: ArtifactReference
    calibration_report: ArtifactReference
    train_claims: ArtifactReference | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Gold-free runtime settings shared by both experiment arms."""

    retrieval_k: int

    def __post_init__(self) -> None:
        if isinstance(self.retrieval_k, bool) or not isinstance(self.retrieval_k, int) or self.retrieval_k <= 0:
            raise ControlledExperimentsConfigError("runtime.retrieval_k must be a positive integer.")


@dataclass(frozen=True, slots=True)
class DirectRAGSettings:
    """Fixed, no-abstention direct-RAG baseline settings.

    SciFact's official abstract score only considers the first three predicted
    rationale sentences.  The baseline therefore emits at most three, ordered
    deterministically by sentence score and then sentence id.
    """

    max_sentences_per_citation: int

    def __post_init__(self) -> None:
        value = self.max_sentences_per_citation
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3:
            raise ControlledExperimentsConfigError(
                "direct_rag.max_sentences_per_citation must be an integer in [1, 3]."
            )


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    """Deterministic paired-bootstrap settings shared by both comparisons."""

    enabled: bool
    resamples: int
    seed: int
    confidence_level: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ControlledExperimentsConfigError("bootstrap.enabled must be a boolean.")
        if isinstance(self.resamples, bool) or not isinstance(self.resamples, int) or self.resamples <= 0:
            raise ControlledExperimentsConfigError("bootstrap.resamples must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ControlledExperimentsConfigError("bootstrap.seed must be a non-negative integer.")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0.0 < self.confidence_level < 1.0
        ):
            raise ControlledExperimentsConfigError("bootstrap.confidence_level must lie in (0, 1).")


@dataclass(frozen=True, slots=True)
class ControlledExperimentOutput:
    """Separate Phase 8 output locations."""

    result_path: Path
    trace_path: Path
    direct_predictions_path: Path
    audited_predictions_path: Path
    report_path: Path
    agent_trace_path: Path

    def __post_init__(self) -> None:
        paths = {
            "result_path": self.result_path,
            "trace_path": self.trace_path,
            "direct_predictions_path": self.direct_predictions_path,
            "audited_predictions_path": self.audited_predictions_path,
            "report_path": self.report_path,
            "agent_trace_path": self.agent_trace_path,
        }
        if len({str(path) for path in paths.values()}) != len(paths):
            raise ControlledExperimentsConfigError("output paths must all be distinct.")


@dataclass(frozen=True, slots=True)
class ControlledExperimentsConfig:
    """Fully path-resolved and schema-validated Phase 8 configuration."""

    config_path: Path
    project_root: Path
    artifacts: ControlledExperimentArtifacts
    runtime: RuntimeSettings
    direct_rag: DirectRAGSettings
    bootstrap: BootstrapSettings
    output: ControlledExperimentOutput


def _resolve_path(config_dir: Path, raw_value: object, field_name: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ControlledExperimentsConfigError(f"{field_name} must be a non-empty string path.")
    candidate = Path(raw_value)
    resolved = candidate if candidate.is_absolute() else config_dir / candidate
    return resolved.resolve()


def _require_mapping(payload: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ControlledExperimentsConfigError(f"{field_name} must be a mapping.")
    return payload


def _parse_artifact_reference(
    config_dir: Path, section: object, field_name: str
) -> ArtifactReference:
    mapping = _require_mapping(section, field_name)
    _require_exact_keys(mapping, {"path", "sha256"}, field_name)
    sha256 = mapping.get("sha256")
    if not isinstance(sha256, str):
        raise ControlledExperimentsConfigError(f"{field_name}.sha256 must be a string.")
    return ArtifactReference(
        path=_resolve_path(config_dir, mapping.get("path"), f"{field_name}.path"),
        sha256=sha256,
    )


def load_controlled_experiments_config(config_path: Path) -> ControlledExperimentsConfig:
    """Load the strict Phase 8 configuration without reading any artifacts."""
    config_path = Path(config_path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ControlledExperimentsConfigError(
            f"Unable to read controlled-experiments config {config_path}: {error}"
        ) from error
    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ControlledExperimentsConfigError(f"Invalid YAML in {config_path}: {error}") from error

    payload = _require_mapping(payload, str(config_path))
    _require_exact_keys(
        payload,
        {"schema_version", "label", "artifacts", "runtime", "direct_rag", "bootstrap", "output"},
        str(config_path),
    )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ControlledExperimentsConfigError(
            f"{config_path} must declare schema_version '{CONFIG_SCHEMA_VERSION}'."
        )
    if payload.get("label") != EVALUATION_LABEL:
        raise ControlledExperimentsConfigError(
            f"{config_path} must declare label '{EVALUATION_LABEL}'."
        )

    config_dir = config_path.resolve().parent
    project_root = config_dir.parent

    artifact_section = _require_mapping(payload.get("artifacts"), "artifacts")
    missing = [name for name in _REQUIRED_ARTIFACTS if name not in artifact_section]
    if missing:
        raise ControlledExperimentsConfigError(
            f"artifacts is missing required entries: {sorted(missing)}"
        )
    unknown = set(artifact_section) - set(_REQUIRED_ARTIFACTS) - set(_OPTIONAL_ARTIFACTS)
    if unknown:
        raise ControlledExperimentsConfigError(f"artifacts has unknown entries: {sorted(unknown)}")
    artifacts = ControlledExperimentArtifacts(
        corpus=_parse_artifact_reference(config_dir, artifact_section["corpus"], "artifacts.corpus"),
        claims_dev=_parse_artifact_reference(
            config_dir, artifact_section["claims_dev"], "artifacts.claims_dev"
        ),
        bm25_index=_parse_artifact_reference(
            config_dir, artifact_section["bm25_index"], "artifacts.bm25_index"
        ),
        verifier_model=_parse_artifact_reference(
            config_dir, artifact_section["verifier_model"], "artifacts.verifier_model"
        ),
        calibration_report=_parse_artifact_reference(
            config_dir, artifact_section["calibration_report"], "artifacts.calibration_report"
        ),
        train_claims=(
            _parse_artifact_reference(
                config_dir, artifact_section["train_claims"], "artifacts.train_claims"
            )
            if "train_claims" in artifact_section
            else None
        ),
    )

    runtime_section = _require_mapping(payload.get("runtime"), "runtime")
    _require_exact_keys(runtime_section, {"retrieval_k"}, "runtime")
    runtime = RuntimeSettings(retrieval_k=runtime_section.get("retrieval_k"))

    direct_section = _require_mapping(payload.get("direct_rag"), "direct_rag")
    _require_exact_keys(direct_section, {"max_sentences_per_citation"}, "direct_rag")
    direct_rag = DirectRAGSettings(
        max_sentences_per_citation=direct_section.get("max_sentences_per_citation")
    )

    bootstrap_section = _require_mapping(payload.get("bootstrap"), "bootstrap")
    _require_exact_keys(
        bootstrap_section,
        {"enabled", "resamples", "seed", "confidence_level"},
        "bootstrap",
    )
    bootstrap = BootstrapSettings(
        enabled=bootstrap_section.get("enabled"),
        resamples=bootstrap_section.get("resamples"),
        seed=bootstrap_section.get("seed"),
        confidence_level=bootstrap_section.get("confidence_level"),
    )

    output_section = _require_mapping(payload.get("output"), "output")
    _require_exact_keys(
        output_section,
        {
            "result_path",
            "trace_path",
            "direct_predictions_path",
            "audited_predictions_path",
            "report_path",
            "agent_trace_path",
        },
        "output",
    )
    output = ControlledExperimentOutput(
        result_path=_resolve_path(config_dir, output_section.get("result_path"), "output.result_path"),
        trace_path=_resolve_path(config_dir, output_section.get("trace_path"), "output.trace_path"),
        direct_predictions_path=_resolve_path(
            config_dir, output_section.get("direct_predictions_path"), "output.direct_predictions_path"
        ),
        audited_predictions_path=_resolve_path(
            config_dir, output_section.get("audited_predictions_path"), "output.audited_predictions_path"
        ),
        report_path=_resolve_path(config_dir, output_section.get("report_path"), "output.report_path"),
        agent_trace_path=_resolve_path(
            config_dir, output_section.get("agent_trace_path"), "output.agent_trace_path"
        ),
    )
    forbidden_paths = {
        (project_root / relative_path).resolve() for relative_path in _FORBIDDEN_OUTPUT_RELATIVE_PATHS
    }
    artifact_paths = {
        artifact.path
        for artifact in (
            artifacts.corpus,
            artifacts.claims_dev,
            artifacts.bm25_index,
            artifacts.verifier_model,
            artifacts.calibration_report,
            artifacts.train_claims,
        )
        if artifact is not None
    }
    for field_name, path in (
        ("output.result_path", output.result_path),
        ("output.trace_path", output.trace_path),
        ("output.direct_predictions_path", output.direct_predictions_path),
        ("output.audited_predictions_path", output.audited_predictions_path),
        ("output.report_path", output.report_path),
        ("output.agent_trace_path", output.agent_trace_path),
    ):
        if path in forbidden_paths:
            raise ControlledExperimentsConfigError(
                f"{field_name} must not reuse a Phase 6 or Phase 7 output path: {path}"
            )
        if path in artifact_paths:
            raise ControlledExperimentsConfigError(
                f"{field_name} must not overwrite a declared input artifact: {path}"
            )

    return ControlledExperimentsConfig(
        config_path=config_path.resolve(),
        project_root=project_root,
        artifacts=artifacts,
        runtime=runtime,
        direct_rag=direct_rag,
        bootstrap=bootstrap,
        output=output,
    )
