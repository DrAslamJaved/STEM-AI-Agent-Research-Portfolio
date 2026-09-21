"""Real benchmark execution utilities for Project 07.

The earlier phases deliberately used synthetic fixtures to test the workflow.
This module is the boundary between that engineering fixture and a genuine,
auditable study:

* miniF2F theorem headers are extracted without their reference proof bodies;
* every arm sees the same frozen task manifest and sampling schedule;
* the LLM-only and SymPy arms close generation before Lean evaluation starts;
* Lean repair may receive only bounded independent compiler diagnostics; and
* every proof metric is derived from a captured Lean result, never prose or
  SymPy output.

No network request is made merely by importing this module.  The optional
``OpenAIResponsesProvider`` reads its key only when a live run is explicitly
started by the command-line runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from formal_math.evaluation_harness import (
    Candidate,
    Compiler,
    CompilerOutcome,
    EvaluationResult,
    LeanSubprocessCompiler,
    Task,
    evaluate_candidate,
)
from formal_math.evaluation_release import ARM_POLICY, build_release_bundle, sha256_file
from formal_math.lean_repair import LeanCompilerFeedback
from formal_math.sympy_counterexamples import SympyContext, find_singleton_jaccard_identity_counterexample


PINNED_MINIF2F_COMMIT = "f0dcc8b59e630fba00ba9569ca6714700e0a8801"
PINNED_LEAN_TOOLCHAIN = "leanprover-community/lean:3.42.1"
PINNED_MATHLIB_REVISION = "cb2b02fff213ed6f65bebd64446baac64137dcda"
EXPECTED_ARMS = ("llm_only", "llm_sympy", "llm_lean_repair")
FROZEN_SPLITS = frozenset({"valid", "test"})
FROZEN_THEOREM_PATTERN = re.compile(
    r"(?ms)^(theorem\s+([A-Za-z_][A-Za-z0-9_']*)\b.*?)(?=^begin\b)"
)
FROZEN_IMPORT_PATTERN = re.compile(r"(?m)^import\s+([A-Za-z0-9_.'-]+)\s*$")
FROZEN_OPEN_LOCALE_PATTERN = re.compile(r"(?m)^open_locale\s+.+$")
FORBIDDEN_PROOF_TOKENS = ("sorry", "admit", "axiom")


def _canonical_json_bytes(payload: Any) -> bytes:
    """Encode JSON deterministically and with LF-only line endings."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_revision(path: Path) -> str:
    """Return the exact Git revision of a local benchmark checkout."""
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot read Git revision for {path}: {completed.stderr.strip()}")
    return completed.stdout.strip()


@dataclass(frozen=True)
class StudyTask:
    """A frozen theorem plus study-only classification and provenance."""

    task: Task
    kind: str
    source_path: str
    source_sha256: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"proof", "refutation"}:
            raise ValueError("kind must be 'proof' or 'refutation'")
        if not self.source_path or len(self.source_sha256) != 64:
            raise ValueError("source_path and source_sha256 are required")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self.task)
        record.update(
            {
                "kind": self.kind,
                "source_path": self.source_path,
                "source_sha256": self.source_sha256,
                "tags": list(self.tags),
            }
        )
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StudyTask":
        tags = record.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            raise ValueError("task tags must be a list of strings")
        return cls(
            task=Task.from_record(record),
            kind=str(record.get("kind", "proof")),
            source_path=str(record.get("source_path", "")),
            source_sha256=str(record.get("source_sha256", "")),
            tags=tuple(tags),
        )


def _select_deterministic_subset(
    tasks: Sequence[StudyTask], *, limit: int | None, selection_seed: int
) -> list[StudyTask]:
    if limit is None or limit == 0 or limit >= len(tasks):
        return list(tasks)
    if limit < 1:
        raise ValueError("limit must be positive or zero for the complete split")
    ranked = sorted(
        enumerate(tasks),
        key=lambda item: sha256(
            f"{selection_seed}|{item[1].task.theorem_id}".encode("utf-8")
        ).hexdigest(),
    )
    selected_indexes = {index for index, _ in ranked[:limit]}
    return [task for index, task in enumerate(tasks) if index in selected_indexes]


def extract_minif2f_tasks(
    mini_f2f_root: Path,
    *,
    split: str,
    limit: int | None = None,
    selection_seed: int = 20260921,
) -> list[StudyTask]:
    """Extract Lean theorem declarations while never carrying over proof bodies.

    miniF2F v1 Lean files use a top-level ``theorem ... := begin`` layout.
    The extraction is intentionally conservative: a format change or a count
    other than 244 stops the study rather than silently producing an altered
    benchmark.
    """
    if split not in FROZEN_SPLITS:
        raise ValueError("split must be 'valid' or 'test'")
    source_path = mini_f2f_root / "lean" / "src" / f"{split}.lean"
    if not source_path.is_file():
        raise FileNotFoundError(f"miniF2F Lean source is missing: {source_path}")
    text = source_path.read_text(encoding="utf-8")
    source_digest = sha256(text.encode("utf-8")).hexdigest()
    imports = tuple(FROZEN_IMPORT_PATTERN.findall(text))
    if imports != ("minif2f_import",):
        raise ValueError("unexpected miniF2F import preamble; inspect the frozen source")
    before_theorems = text[: text.find("theorem ")]
    preamble = "\n".join(FROZEN_OPEN_LOCALE_PATTERN.findall(before_theorems))
    if not preamble:
        raise ValueError("miniF2F locale preamble is unexpectedly absent")

    extracted: list[StudyTask] = []
    for match in FROZEN_THEOREM_PATTERN.finditer(text):
        declaration = match.group(1).rstrip()
        theorem_name = match.group(2)
        if not declaration.endswith(":="):
            raise ValueError(f"could not isolate declaration for {theorem_name}")
        extracted.append(
            StudyTask(
                task=Task(
                    theorem_id=f"minif2f_{split}_{theorem_name}",
                    split=split,
                    declaration=declaration,
                    imports=imports,
                    preamble=preamble,
                    source="miniF2F_v1",
                ),
                kind="proof",
                source_path=f"lean/src/{split}.lean",
                source_sha256=source_digest,
                tags=("miniF2F", split),
            )
        )
    if len(extracted) != 244:
        raise ValueError(
            f"expected exactly 244 miniF2F {split} theorems, extracted {len(extracted)}"
        )
    return _select_deterministic_subset(
        extracted, limit=limit, selection_seed=selection_seed
    )


def build_fuzzy_study_tasks(project_root: Path, *, split: str) -> list[StudyTask]:
    """Return an executable finite-grid fuzzy proof/refutation suite.

    The preamble contains definitions only; it intentionally excludes the
    already-proved Phase 07 theorems so candidate generators cannot simply
    invoke a preloaded answer.
    """
    if split not in FROZEN_SPLITS:
        raise ValueError("split must be 'valid' or 'test'")
    source_path = project_root / "formalizations" / "lean3" / "FuzzySimilarity.lean"
    if not source_path.is_file():
        raise FileNotFoundError(f"fuzzy formalization is missing: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    marker = "/- The explicit empty-union convention"
    try:
        definition_source = source[: source.index(marker)]
    except ValueError as error:
        raise ValueError("fuzzy formalization no longer has the registered kernel marker") from error
    kernel = definition_source.rstrip() + "\n\nend project07\n\nopen project07"
    digest = sha256(source.encode("utf-8")).hexdigest()
    common = {
        "split": split,
        "imports": (),
        "preamble": kernel,
        "source": "project07_fuzzy_suite",
    }
    definitions = (
        (
            "project07_fuzzy_zero_zero",
            "proof",
            "theorem project07_fuzzy_zero_zero :\n"
            "  fuzzy_jaccard fuzzy_member.zero fuzzy_member.zero = similarity_value.one :=",
            ("fuzzy", "empty_union_convention"),
        ),
        (
            "project07_fuzzy_refl_quarter",
            "proof",
            "theorem project07_fuzzy_refl_quarter :\n"
            "  fuzzy_jaccard fuzzy_member.quarter fuzzy_member.quarter = similarity_value.one :=",
            ("fuzzy", "reflexivity"),
        ),
        (
            "project07_fuzzy_symmetry",
            "proof",
            "theorem project07_fuzzy_symmetry (a b : fuzzy_member) :\n"
            "  fuzzy_jaccard a b = fuzzy_jaccard b a :=",
            ("fuzzy", "symmetry"),
        ),
        (
            "project07_fuzzy_refutation",
            "refutation",
            "theorem project07_fuzzy_refutation :\n"
            "  fuzzy_jaccard fuzzy_member.quarter fuzzy_member.half = similarity_value.one -> false :=",
            ("fuzzy", "counterexample", "refutation"),
        ),
    )
    return [
        StudyTask(
            task=Task(theorem_id=task_id, declaration=declaration, **common),
            kind=kind,
            source_path="formalizations/lean3/FuzzySimilarity.lean",
            source_sha256=digest,
            tags=tags,
        )
        for task_id, kind, declaration, tags in definitions
    ]


def write_task_manifest(path: Path, tasks: Sequence[StudyTask]) -> None:
    """Persist a deterministic LF-only JSONL task manifest."""
    if not tasks:
        raise ValueError("cannot write an empty study task manifest")
    lines = [json.dumps(task.to_record(), sort_keys=True, separators=(",", ":")) for task in tasks]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def load_task_manifest(path: Path) -> list[StudyTask]:
    """Load a study task manifest and reject malformed or mixed-split input."""
    tasks: list[StudyTask] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"task record on line {line_number} must be an object")
        tasks.append(StudyTask.from_record(record))
    if not tasks:
        raise ValueError("task manifest contains no tasks")
    splits = {task.task.split for task in tasks}
    if len(splits) != 1:
        raise ValueError("a study task manifest must contain exactly one split")
    ids = [task.task.theorem_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("study task manifest contains duplicate theorem IDs")
    return tasks


@dataclass(frozen=True)
class Pricing:
    """User-supplied, dated model pricing used only for transparent estimates."""

    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float

    def __post_init__(self) -> None:
        if self.input_usd_per_million_tokens < 0 or self.output_usd_per_million_tokens < 0:
            raise ValueError("token prices must be non-negative")

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            (input_tokens * self.input_usd_per_million_tokens
             + output_tokens * self.output_usd_per_million_tokens)
            / 1_000_000,
            8,
        )


@dataclass(frozen=True)
class ProviderCompletion:
    """A provider response with the provenance needed for the evidence record."""

    text: str
    model_identifier: str
    input_tokens: int
    output_tokens: int
    response_id: str | None = None


class StudyCompletionProvider(Protocol):
    def complete(self, *, prompt: str, seed: int | None, request_label: str) -> ProviderCompletion:
        """Generate one proof body without receiving unauthorised tool evidence."""


def _response_text(payload: Mapping[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    fragments: list[str] = []
    output = payload.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    fragments.append(text)
    rendered = "\n".join(fragment for fragment in fragments if fragment.strip()).strip()
    if not rendered:
        raise RuntimeError("provider response did not contain output text")
    return rendered


class OpenAIResponsesProvider:
    """Small standard-library adapter for an explicitly requested OpenAI run.

    It avoids persisting the API key or the full HTTP response.  The study
    record retains only the raw model completion, response id, model name, and
    reported token usage.
    """

    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        *,
        model_identifier: str,
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: int = 1_024,
        temperature: float | None = 0.0,
        timeout_seconds: float = 120.0,
        send_seed: bool = False,
        transport: Callable[..., Any] | None = None,
    ) -> None:
        if not model_identifier:
            raise ValueError("model_identifier is required")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model_identifier = model_identifier
        self.api_key_env = api_key_env
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.send_seed = send_seed
        self._transport = transport or urlopen

    def complete(self, *, prompt: str, seed: int | None, request_label: str) -> ProviderCompletion:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {self.api_key_env} is required for a live model run")
        payload: dict[str, Any] = {
            "model": self.model_identifier,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": self.max_output_tokens,
            "metadata": {"project": "project_07", "request_label": request_label},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.send_seed and seed is not None:
            payload["seed"] = seed
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._transport(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1_000]
            raise RuntimeError(f"OpenAI request failed with HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"OpenAI request failed: {error.reason}") from error
        if not isinstance(decoded, Mapping):
            raise RuntimeError("OpenAI response is not a JSON object")
        usage = decoded.get("usage", {})
        if not isinstance(usage, Mapping):
            raise RuntimeError("OpenAI response did not include token usage")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            raise RuntimeError("OpenAI response did not include integer input/output token usage")
        returned_model = decoded.get("model")
        return ProviderCompletion(
            text=_response_text(decoded),
            model_identifier=str(returned_model or self.model_identifier),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_id=str(decoded["id"]) if decoded.get("id") is not None else None,
        )


@dataclass(frozen=True)
class LiveStudyPlan:
    """Immutable run configuration written before any live candidate is generated."""

    run_id: str
    evaluation_mode: str
    model_identifier: str
    samples_per_theorem: int
    base_seed: int
    max_repair_iterations: int
    task_manifest_sha256: str
    benchmark_commit: str
    lean_toolchain: str
    mathlib_revision: str
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    maximum_estimated_cost_usd: float
    send_seed_to_provider: bool
    synthetic_demo: bool = False

    def __post_init__(self) -> None:
        if self.evaluation_mode not in {"development", "final_evaluation"}:
            raise ValueError("evaluation_mode must be 'development' or 'final_evaluation'")
        if not self.run_id or not self.model_identifier:
            raise ValueError("run_id and model_identifier are required")
        if self.samples_per_theorem < 1:
            raise ValueError("samples_per_theorem must be positive")
        if self.max_repair_iterations != 2:
            raise ValueError("the registered Lean repair budget is exactly two iterations")
        if len(self.task_manifest_sha256) != 64:
            raise ValueError("task_manifest_sha256 must be a SHA-256 digest")
        if self.synthetic_demo:
            raise ValueError("Phase 09 is reserved for non-synthetic evidence")
        if self.maximum_estimated_cost_usd <= 0:
            raise ValueError("maximum_estimated_cost_usd must be positive")
        Pricing(self.input_usd_per_million_tokens, self.output_usd_per_million_tokens)

    @property
    def pricing(self) -> Pricing:
        return Pricing(self.input_usd_per_million_tokens, self.output_usd_per_million_tokens)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema_version"] = 1
        record["phase"] = "09"
        record["arms"] = {arm: dict(policy) for arm, policy in ARM_POLICY.items()}
        return record


def validate_plan_for_tasks(plan: LiveStudyPlan, tasks: Sequence[StudyTask]) -> None:
    """Enforce split locking and pinned provenance before provider calls."""
    if not tasks:
        raise ValueError("study plan requires at least one task")
    splits = {item.task.split for item in tasks}
    if len(splits) != 1:
        raise ValueError("study plan requires exactly one task split")
    split = next(iter(splits))
    expected_split = "valid" if plan.evaluation_mode == "development" else "test"
    if split != expected_split:
        raise PermissionError(f"{plan.evaluation_mode} requires only '{expected_split}' tasks")
    if plan.benchmark_commit != PINNED_MINIF2F_COMMIT:
        raise ValueError("study plan benchmark commit differs from the frozen project pin")
    if plan.lean_toolchain != PINNED_LEAN_TOOLCHAIN:
        raise ValueError("study plan Lean toolchain differs from the frozen project pin")
    if plan.mathlib_revision != PINNED_MATHLIB_REVISION:
        raise ValueError("study plan mathlib revision differs from the frozen project pin")


def write_study_plan(path: Path, plan: LiveStudyPlan) -> None:
    """Write the pre-generation plan with platform-stable bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(plan.to_record()))


def load_study_plan(path: Path) -> LiveStudyPlan:
    """Load an immutable Phase 09 plan, excluding derived display fields."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("study plan must be a JSON object")
    if payload.get("schema_version") != 1 or payload.get("phase") != "09":
        raise ValueError("not a Phase 09 study plan")
    expected_arms = {arm: dict(policy) for arm, policy in ARM_POLICY.items()}
    if payload.get("arms") != expected_arms:
        raise ValueError("study plan arm policy differs from the registered protocol")
    fields = {
        key: value
        for key, value in payload.items()
        if key not in {"schema_version", "phase", "arms"}
    }
    return LiveStudyPlan(**fields)


def _task_context(task: Task) -> str:
    return task.render_context()


def build_baseline_prompt(task: Task) -> str:
    return (
        "Produce only the Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "You have no tools and will receive no verification feedback.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{_task_context(task)}\n"
    )


def build_sympy_prompt(task: Task, context: SympyContext) -> str:
    return (
        "Produce only the Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "You may use only the recorded SymPy context below; no Lean compiler\n"
        "diagnostics, compilation status, or prior candidate text is available.\n"
        "SymPy evidence is advisory and cannot establish formal validity.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{context.render_for_prompt()}\n\n"
        f"{_task_context(task)}\n"
    )


def build_lean_initial_prompt(task: Task) -> str:
    return (
        "Produce only the Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "This is the initial attempt; no compiler feedback is available yet.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{_task_context(task)}\n"
    )


def build_lean_repair_prompt(
    task: Task, feedback: LeanCompilerFeedback, *, repair_iteration: int, feedback_character_limit: int = 6_000
) -> str:
    if repair_iteration not in {1, 2}:
        raise ValueError("repair_iteration must be one or two")
    rendered_feedback = feedback.render_for_prompt()
    if len(rendered_feedback) > feedback_character_limit:
        rendered_feedback = (
            rendered_feedback[:feedback_character_limit]
            + "\n[diagnostics truncated to the registered feedback limit]"
        )
    return (
        "Produce only a replacement Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "Use only the independent Lean feedback from the immediately prior proof.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{rendered_feedback}\n\n"
        f"Frozen theorem (repair iteration {repair_iteration}):\n{_task_context(task)}\n"
    )


def normalise_proof_body(raw_completion: str) -> tuple[str, str | None]:
    """Keep a model response as a Lean body and preserve malformed output as failure."""
    text = raw_completion.strip()
    fenced = re.search(r"```(?:lean)?\s*\n?(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    by_match = re.search(r"\bby(?:\s|$)", text)
    if by_match and by_match.start() > 0:
        text = text[by_match.start():]
    if text:
        return text, None
    # A comment-only body guarantees a Lean parse failure without fabricating a
    # proof or inserting a prohibited shortcut on the model's behalf.
    return "-- PROJECT07_EMPTY_MODEL_COMPLETION", "empty_model_completion"


def _candidate_id(run_id: str, arm: str, theorem_id: str, sample_index: int, iteration: int) -> str:
    digest = sha256(
        f"{run_id}|{arm}|{theorem_id}|{sample_index}|{iteration}".encode("utf-8")
    ).hexdigest()[:20]
    return f"p07-{arm}-{digest}"


@dataclass
class CostLedger:
    pricing: Pricing
    maximum_estimated_cost_usd: float
    total_estimated_cost_usd: float = 0.0
    request_count: int = 0

    def register(self, completion: ProviderCompletion) -> float:
        cost = self.pricing.estimate(completion.input_tokens, completion.output_tokens)
        self.total_estimated_cost_usd = round(self.total_estimated_cost_usd + cost, 8)
        self.request_count += 1
        if self.total_estimated_cost_usd > self.maximum_estimated_cost_usd:
            raise RuntimeError(
                "live-study cost cap exceeded after the completed request: "
                f"${self.total_estimated_cost_usd:.6f} > ${self.maximum_estimated_cost_usd:.6f}"
            )
        return cost


def _record_attempt(
    *,
    study_task: StudyTask,
    arm: str,
    run_id: str,
    sample_index: int,
    iteration: int,
    prompt: str,
    completion: ProviderCompletion,
    compiler: Compiler,
    estimated_cost_usd: float,
    generation_elapsed_seconds: float,
    seed: int | None,
    sympy_context: SympyContext | None = None,
    feedback: LeanCompilerFeedback | None = None,
) -> tuple[dict[str, Any], Candidate, EvaluationResult]:
    lean_code, normalisation_error = normalise_proof_body(completion.text)
    metadata: dict[str, Any] = {
        "generation_policy": arm,
        "model_identifier": completion.model_identifier,
        "sample_index": sample_index,
        "iteration": iteration,
        "seed": seed,
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_completion_sha256": sha256(completion.text.encode("utf-8")).hexdigest(),
        "lean_feedback": "forbidden" if arm != "llm_lean_repair" else (
            "none" if feedback is None else "compiler_diagnostics_and_static_policy"
        ),
        "sympy_feedback": "allowed" if arm == "llm_sympy" else "forbidden",
    }
    if normalisation_error:
        metadata["normalisation_error"] = normalisation_error
    if sympy_context is not None:
        metadata["sympy_context_sha256"] = sympy_context.sha256
    if feedback is not None:
        metadata["prior_candidate_id"] = feedback.candidate_id
        metadata["lean_feedback_sha256"] = feedback.sha256
    candidate = Candidate(
        candidate_id=_candidate_id(run_id, arm, study_task.task.theorem_id, sample_index, iteration),
        theorem_id=study_task.task.theorem_id,
        arm=arm,
        lean_code=lean_code,
        prompt_id="phase_09_live_study_prompt_v1",
        iteration=iteration,
        metadata=metadata,
    )
    evaluation = evaluate_candidate(study_task.task, candidate, compiler)
    record = {
        "candidate": candidate.to_record(),
        "evaluation": evaluation.to_record(),
        "kind": study_task.kind,
        "sample_index": sample_index,
        "iteration": iteration,
        "prompt": prompt,
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_completion": completion.text,
        "raw_completion_sha256": sha256(completion.text.encode("utf-8")).hexdigest(),
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "generation_elapsed_seconds": generation_elapsed_seconds,
        "human_intervention": False,
        "provider_response_id": completion.response_id,
        "source_provenance": {
            "task_source_path": study_task.source_path,
            "task_source_sha256": study_task.source_sha256,
        },
        "sympy_context": None if sympy_context is None else sympy_context.to_record(),
        "repair_feedback": None if feedback is None else feedback.to_record(),
    }
    return record, candidate, evaluation


def sympy_context_for_task(study_task: StudyTask) -> SympyContext:
    """Supply exact SymPy evidence only for the registered fuzzy refutation."""
    if study_task.task.theorem_id == "project07_fuzzy_refutation":
        return SympyContext(counterexamples=(find_singleton_jaccard_identity_counterexample(),))
    return SympyContext()


def run_live_study(
    tasks: Sequence[StudyTask],
    *,
    provider: StudyCompletionProvider,
    compiler: Compiler,
    plan: LiveStudyPlan,
    generation_journal_path: Path | None = None,
) -> tuple[list[dict[str, Any]], CostLedger]:
    """Run all three registered arms under the frozen protocol.

    Baseline and SymPy candidates are generated as closed batches before their
    first Lean evaluation.  This preserves the no-Lean-feedback requirement.
    Lean repair is intentionally evaluated online because that arm alone is
    allowed to feed diagnostics into a bounded repair loop.
    """
    validate_plan_for_tasks(plan, tasks)
    ledger = CostLedger(plan.pricing, plan.maximum_estimated_cost_usd)
    records: list[dict[str, Any]] = []
    resolved_model_identifier: str | None = None

    def append_generation_journal(
        *, prompt: str, seed: int | None, request_label: str, completion: ProviderCompletion
    ) -> None:
        if generation_journal_path is None:
            return
        generation_journal_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": 1,
            "phase": "09",
            "event": "completion_captured_before_evaluation",
            "captured_at_utc": _utc_now(),
            "request_label": request_label,
            "seed": seed,
            "prompt": prompt,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_completion": completion.text,
            "raw_completion_sha256": sha256(completion.text.encode("utf-8")).hexdigest(),
            "model_identifier": completion.model_identifier,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "provider_response_id": completion.response_id,
        }
        with generation_journal_path.open("ab") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            handle.write(b"\n")

    def complete(
        *, prompt: str, seed: int | None, request_label: str
    ) -> tuple[ProviderCompletion, float, float]:
        nonlocal resolved_model_identifier
        started = perf_counter()
        completion = provider.complete(prompt=prompt, seed=seed, request_label=request_label)
        generation_elapsed = round(perf_counter() - started, 6)
        if resolved_model_identifier is None:
            resolved_model_identifier = completion.model_identifier
        elif completion.model_identifier != resolved_model_identifier:
            raise RuntimeError(
                "provider changed its resolved model identifier during one controlled run: "
                f"{completion.model_identifier!r} != {resolved_model_identifier!r}"
            )
        append_generation_journal(
            prompt=prompt,
            seed=seed,
            request_label=request_label,
            completion=completion,
        )
        return completion, ledger.register(completion), generation_elapsed

    baseline_queue: list[tuple[StudyTask, int, int, str, ProviderCompletion, float, float]] = []
    for study_task in tasks:
        for sample_index in range(1, plan.samples_per_theorem + 1):
            prompt = build_baseline_prompt(study_task.task)
            seed = plan.base_seed + sample_index - 1
            completion, cost, generation_elapsed = complete(
                prompt=prompt,
                seed=seed,
                request_label=f"{plan.run_id}:llm_only:{study_task.task.theorem_id}:{sample_index}",
            )
            baseline_queue.append((study_task, sample_index, seed, prompt, completion, cost, generation_elapsed))
    for study_task, sample_index, seed, prompt, completion, cost, generation_elapsed in baseline_queue:
        record, _, _ = _record_attempt(
            study_task=study_task,
            arm="llm_only",
            run_id=plan.run_id,
            sample_index=sample_index,
            iteration=0,
            prompt=prompt,
            completion=completion,
            compiler=compiler,
            estimated_cost_usd=cost,
            generation_elapsed_seconds=generation_elapsed,
            seed=seed,
        )
        records.append(record)

    sympy_queue: list[tuple[StudyTask, int, int, SympyContext, str, ProviderCompletion, float, float]] = []
    for study_task in tasks:
        context = sympy_context_for_task(study_task)
        for sample_index in range(1, plan.samples_per_theorem + 1):
            prompt = build_sympy_prompt(study_task.task, context)
            seed = plan.base_seed + sample_index - 1
            completion, cost, generation_elapsed = complete(
                prompt=prompt,
                seed=seed,
                request_label=f"{plan.run_id}:llm_sympy:{study_task.task.theorem_id}:{sample_index}",
            )
            sympy_queue.append(
                (study_task, sample_index, seed, context, prompt, completion, cost, generation_elapsed)
            )
    for study_task, sample_index, seed, context, prompt, completion, cost, generation_elapsed in sympy_queue:
        record, _, _ = _record_attempt(
            study_task=study_task,
            arm="llm_sympy",
            run_id=plan.run_id,
            sample_index=sample_index,
            iteration=0,
            prompt=prompt,
            completion=completion,
            compiler=compiler,
            estimated_cost_usd=cost,
            generation_elapsed_seconds=generation_elapsed,
            seed=seed,
            sympy_context=context,
        )
        records.append(record)

    for study_task in tasks:
        for sample_index in range(1, plan.samples_per_theorem + 1):
            prompt = build_lean_initial_prompt(study_task.task)
            seed = plan.base_seed + sample_index - 1
            completion, cost, generation_elapsed = complete(
                prompt=prompt,
                seed=seed,
                request_label=f"{plan.run_id}:llm_lean_repair:{study_task.task.theorem_id}:{sample_index}:0",
            )
            record, candidate, evaluation = _record_attempt(
                study_task=study_task,
                arm="llm_lean_repair",
                run_id=plan.run_id,
                sample_index=sample_index,
                iteration=0,
                prompt=prompt,
                completion=completion,
                compiler=compiler,
                estimated_cost_usd=cost,
                generation_elapsed_seconds=generation_elapsed,
                seed=seed,
            )
            records.append(record)
            for repair_iteration in range(1, plan.max_repair_iterations + 1):
                if evaluation.compile_passed:
                    break
                feedback = LeanCompilerFeedback.from_evaluation(candidate, evaluation)
                repair_prompt = build_lean_repair_prompt(
                    study_task.task, feedback, repair_iteration=repair_iteration
                )
                repair_seed = plan.base_seed + 10_000 * sample_index + repair_iteration
                repair_completion, repair_cost, repair_generation_elapsed = complete(
                    prompt=repair_prompt,
                    seed=repair_seed,
                    request_label=(
                        f"{plan.run_id}:llm_lean_repair:{study_task.task.theorem_id}:"
                        f"{sample_index}:{repair_iteration}"
                    ),
                )
                record, candidate, evaluation = _record_attempt(
                    study_task=study_task,
                    arm="llm_lean_repair",
                    run_id=plan.run_id,
                    sample_index=sample_index,
                    iteration=repair_iteration,
                    prompt=repair_prompt,
                    completion=repair_completion,
                    compiler=compiler,
                    estimated_cost_usd=repair_cost,
                    generation_elapsed_seconds=repair_generation_elapsed,
                    seed=repair_seed,
                    feedback=feedback,
                )
                records.append(record)
    return records, ledger


def write_attempt_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write immutable raw attempt records with LF-only JSONL bytes."""
    if not records:
        raise ValueError("cannot write an empty attempt record set")
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def final_release_manifest(
    *,
    plan: LiveStudyPlan,
    tasks_path: Path,
    attempts_path: Path,
    reviewer: str = "pending_human_review",
) -> dict[str, Any]:
    """Create the Phase 08-compatible evidence manifest for a real run."""
    if plan.evaluation_mode != "final_evaluation":
        raise PermissionError("a Phase 08 release manifest can be created only from the locked test split")
    return {
        "schema_version": 1,
        "phase": "08",
        "run_id": plan.run_id,
        "evaluation_split": "test",
        "test_split_locked": True,
        "formal_validity_decision": "independent_lean_compilation",
        "samples_per_theorem": plan.samples_per_theorem,
        "max_repair_iterations": plan.max_repair_iterations,
        "model_identifier": plan.model_identifier,
        "seed": plan.base_seed,
        "synthetic_demo": False,
        "arms": {arm: dict(policy) for arm, policy in ARM_POLICY.items()},
        "provenance": {
            "benchmark_commit": plan.benchmark_commit,
            "lean_toolchain": plan.lean_toolchain,
            "mathlib_revision": plan.mathlib_revision,
            "task_manifest_sha256": sha256_file(tasks_path),
            "candidate_records_sha256": sha256_file(attempts_path),
        },
        "human_review": {"approved": False, "reviewer": reviewer},
    }


def write_real_run_bundle(
    output_directory: Path,
    *,
    plan: LiveStudyPlan,
    tasks_path: Path,
    records: Sequence[Mapping[str, Any]],
    ledger: CostLedger,
) -> dict[str, Path]:
    """Persist real evidence and a deliberately blocked pre-review release bundle.

    A development run is useful for tuning the protocol but cannot be pushed
    through the Phase 08 final-release machinery.  It receives a separate
    clearly labelled development summary instead.
    """
    if output_directory.exists():
        existing = {path.name for path in output_directory.iterdir()}
        if existing - {"generation_journal.jsonl"}:
            raise FileExistsError(
                "refusing to write a bundle into a non-empty run directory: "
                f"{output_directory}"
            )
    else:
        output_directory.mkdir(parents=True)
    attempts_path = output_directory / "candidate_attempts.jsonl"
    write_attempt_records(attempts_path, records)
    artifacts: dict[str, Path] = {"attempts": attempts_path}
    if plan.evaluation_mode == "final_evaluation":
        manifest = final_release_manifest(plan=plan, tasks_path=tasks_path, attempts_path=attempts_path)
        manifest_path = output_directory / "locked_evaluation_manifest.json"
        manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")

        from formal_math.evaluation_release import (
            load_attempts_jsonl,
            write_release_bundle,
            write_release_report,
        )

        bundle = build_release_bundle(manifest, load_attempts_jsonl(attempts_path))
        evidence_path = output_directory / "release_evidence.json"
        report_path = output_directory / "release_report.md"
        write_release_bundle(evidence_path, bundle)
        write_release_report(report_path, bundle)
        artifacts.update(
            {
                "manifest": manifest_path,
                "evidence": evidence_path,
                "report": report_path,
            }
        )
        release_ready = bundle["release_gate"]["release_ready"]
    else:
        from formal_math.evaluation_release import calculate_arm_metrics, load_attempts_jsonl

        attempts = load_attempts_jsonl(attempts_path)
        evidence_path = output_directory / "development_evidence.json"
        evidence = {
            "schema_version": 1,
            "phase": "09",
            "evaluation_split": "valid",
            "synthetic_demo": False,
            "release_ready": False,
            "release_blocker": "development split is never final evidence",
            "formal_validity_decision": "independent_lean_compilation",
            "arm_metrics": calculate_arm_metrics(
                attempts, samples_per_theorem=plan.samples_per_theorem
            ),
        }
        evidence_path.write_bytes(json.dumps(evidence, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        artifacts["development_evidence"] = evidence_path
        release_ready = False
    metadata_path = output_directory / "execution_metadata.json"
    metadata = {
        "schema_version": 1,
        "phase": "09",
        "created_at_utc": _utc_now(),
        "synthetic_demo": False,
        "plan": plan.to_record(),
        "task_manifest_path": str(tasks_path),
        "task_manifest_sha256": sha256_file(tasks_path),
        "candidate_records_sha256": sha256_file(attempts_path),
        "request_count": ledger.request_count,
        "total_estimated_cost_usd": round(ledger.total_estimated_cost_usd, 8),
        "maximum_estimated_cost_usd": plan.maximum_estimated_cost_usd,
        "release_ready_before_human_review": release_ready,
    }
    metadata_path.write_bytes(json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    artifacts["metadata"] = metadata_path
    return artifacts


def make_pinned_live_compiler(elan_executable: Path, *, cwd: Path, timeout_seconds: float) -> LeanSubprocessCompiler:
    """Construct the independent compiler used by a real three-arm run."""
    if not elan_executable.is_file():
        raise FileNotFoundError(f"elan executable is missing: {elan_executable}")
    if not (cwd / "leanpkg.toml").is_file():
        raise FileNotFoundError(f"configured miniF2F checkout is missing leanpkg.toml: {cwd}")
    if git_revision(cwd) != PINNED_MINIF2F_COMMIT:
        raise ValueError("miniF2F checkout does not match the frozen Project 07 commit")
    return LeanSubprocessCompiler(
        [str(elan_executable), "run", PINNED_LEAN_TOOLCHAIN, "lean"],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
