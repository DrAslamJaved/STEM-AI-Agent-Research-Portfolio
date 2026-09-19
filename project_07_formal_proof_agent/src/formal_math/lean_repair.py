"""Bounded Lean generate-verify-repair primitives for Project 07 Phase 06.

Unlike the LLM-only and LLM+SymPy arms, this arm is explicitly allowed to feed
independent Lean compiler diagnostics back into a bounded repair loop.  The
frozen theorem declaration never changes; every attempt is captured with its
evaluation result and provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence

from formal_math.evaluation_harness import (
    Candidate,
    Compiler,
    EvaluationResult,
    LeanSubprocessCompiler,
    Task,
    assert_split_is_allowed,
    evaluate_candidate,
)


PINNED_MINIF2F_COMMIT = "f0dcc8b59e630fba00ba9569ca6714700e0a8801"
PINNED_LEAN_TOOLCHAIN = "leanprover-community/lean:3.42.1"
PINNED_MATHLIB_REVISION = "cb2b02fff213ed6f65bebd64446baac64137dcda"
LEAN_REPAIR_POLICY = {
    "arm": "llm_lean_repair",
    "lean_feedback_during_generation": "allowed_compiler_diagnostics_and_static_policy",
    "theorem_declaration_edits": "forbidden",
    "max_repair_iterations": 2,
    "development_split": "valid",
    "test_split_locked": True,
    "formal_validity_decision": "independent_lean_compilation",
}
_FENCED_BLOCK = re.compile(r"```(?:lean)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LeanRepairRunConfig:
    """Pre-registered metadata and repair budget for one controlled run."""

    run_id: str
    model_identifier: str
    prompt_id: str
    max_repair_iterations: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.model_identifier or not self.prompt_id:
            raise ValueError("run_id, model_identifier, and prompt_id are required")
        if self.max_repair_iterations < 0:
            raise ValueError("max_repair_iterations must be non-negative")
        if self.max_repair_iterations > LEAN_REPAIR_POLICY["max_repair_iterations"]:
            raise ValueError("max_repair_iterations exceeds the registered Phase 06 budget")


@dataclass(frozen=True)
class InitialGenerationRequest:
    """Initial provider input contains only the frozen theorem prompt."""

    theorem_id: str
    prompt: str
    model_identifier: str
    seed: int | None


@dataclass(frozen=True)
class LeanCompilerFeedback:
    """The independent evaluator's result supplied to a permitted repair step."""

    candidate_id: str
    prior_lean_code: str
    status: str
    compiler_exit_code: int | None
    compiler_stdout: str
    compiler_stderr: str
    static_rejections: tuple[str, ...]

    @classmethod
    def from_evaluation(cls, candidate: Candidate, result: EvaluationResult) -> "LeanCompilerFeedback":
        if candidate.candidate_id != result.candidate_id:
            raise ValueError("candidate and evaluation result do not match")
        return cls(
            candidate_id=candidate.candidate_id,
            prior_lean_code=candidate.lean_code,
            status=result.status,
            compiler_exit_code=result.compiler_exit_code,
            compiler_stdout=result.compiler_stdout,
            compiler_stderr=result.compiler_stderr,
            static_rejections=result.static_rejections,
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_record(), sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()

    def render_for_prompt(self) -> str:
        """Render compiler/static evidence without exposing a mutable theorem."""
        lines = [
            "Independent evaluation feedback for the previous proof body:",
            f"- Previous candidate ID: {self.candidate_id}",
            f"- Evaluation status: {self.status}",
            "- Previous Lean proof body:",
            self.prior_lean_code,
        ]
        if self.static_rejections:
            lines.append(f"- Static policy rejection(s): {', '.join(self.static_rejections)}")
        if self.compiler_exit_code is not None:
            lines.append(f"- Lean exit code: {self.compiler_exit_code}")
        if self.compiler_stdout:
            lines.append(f"- Lean stdout:\n{self.compiler_stdout}")
        if self.compiler_stderr:
            lines.append(f"- Lean stderr:\n{self.compiler_stderr}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LeanRepairRequest:
    """Repair provider input: frozen theorem plus prior code and Lean feedback."""

    theorem_id: str
    prompt: str
    feedback: LeanCompilerFeedback
    model_identifier: str
    seed: int | None
    repair_iteration: int


@dataclass(frozen=True)
class LeanCompletion:
    """Raw provider completion captured before independent evaluation."""

    text: str
    model_identifier: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LeanRepairProvider(Protocol):
    """Interface separates initial generation from a compiler-informed repair."""

    def complete_initial(self, request: InitialGenerationRequest) -> LeanCompletion:
        ...

    def complete_repair(self, request: LeanRepairRequest) -> LeanCompletion:
        ...


@dataclass(frozen=True)
class LeanRepairAttempt:
    """One candidate attempt and the evaluation result that governs continuation."""

    candidate: Candidate
    prompt: str
    raw_completion: str
    evaluation: EvaluationResult
    feedback: LeanCompilerFeedback | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_record(self) -> dict[str, Any]:
        record = self.candidate.to_record()
        record.update(
            {
                "prompt": self.prompt,
                "prompt_sha256": sha256(self.prompt.encode("utf-8")).hexdigest(),
                "raw_completion": self.raw_completion,
                "raw_completion_sha256": sha256(self.raw_completion.encode("utf-8")).hexdigest(),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "evaluation": self.evaluation.to_record(),
                "repair_feedback": None if self.feedback is None else self.feedback.to_record(),
            }
        )
        return record


def build_initial_prompt(task: Task) -> str:
    """Return a frozen-theorem initial prompt with no evaluation feedback."""
    imports = "\n".join(f"import {module}" for module in task.imports)
    frozen_task = "\n\n".join(part for part in (imports, task.declaration.strip()) if part)
    return (
        "Produce only the Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "This is the initial attempt; no compiler feedback is available yet.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{frozen_task}\n"
    )


def build_repair_prompt(task: Task, feedback: LeanCompilerFeedback, repair_iteration: int) -> str:
    """Return a repair prompt containing an immutable theorem and recorded feedback."""
    if repair_iteration < 1:
        raise ValueError("repair_iteration must be at least one")
    imports = "\n".join(f"import {module}" for module in task.imports)
    frozen_task = "\n\n".join(part for part in (imports, task.declaration.strip()) if part)
    return (
        "Produce only a replacement Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "Use the independent Lean feedback to repair the previous proof body.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{feedback.render_for_prompt()}\n\n"
        f"Frozen theorem (Lean repair iteration {repair_iteration}):\n{frozen_task}\n"
    )


def normalise_proof_body(raw_completion: str) -> str:
    """Remove a Markdown fence and preserve the proof body supplied by a provider."""
    text = raw_completion.strip()
    fenced = _FENCED_BLOCK.search(text)
    if fenced:
        text = fenced.group(1).strip()
    by_match = re.search(r"\bby(?:\s|$)", text)
    if by_match and by_match.start() > 0:
        text = text[by_match.start():]
    if not text:
        raise ValueError("completion did not contain a Lean proof body")
    return text


def make_pinned_lean3_compiler(
    elan_executable: Path,
    *,
    cwd: Path,
    timeout_seconds: float = 60.0,
) -> LeanSubprocessCompiler:
    """Create the independent compiler for the Phase 02 pinned Lean 3 toolchain."""
    if not elan_executable.is_file():
        raise FileNotFoundError(f"elan executable not found: {elan_executable}")
    if not cwd.is_dir():
        raise FileNotFoundError(f"Lean working directory not found: {cwd}")
    return LeanSubprocessCompiler(
        [str(elan_executable), "run", PINNED_LEAN_TOOLCHAIN, "lean"],
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )


def _candidate_id(run_id: str, theorem_id: str, iteration: int) -> str:
    material = f"{run_id}|{theorem_id}|{iteration}".encode("utf-8")
    return f"lean-repair-{sha256(material).hexdigest()[:16]}"


def _capture_attempt(
    *,
    task: Task,
    config: LeanRepairRunConfig,
    iteration: int,
    prompt: str,
    completion: LeanCompletion,
    compiler: Compiler,
    feedback: LeanCompilerFeedback | None,
) -> LeanRepairAttempt:
    raw_completion = completion.text
    lean_code = normalise_proof_body(raw_completion)
    model_identifier = completion.model_identifier or config.model_identifier
    metadata: dict[str, Any] = {
        "generation_policy": "bounded_lean_generate_verify_repair",
        "model_identifier": model_identifier,
        "repair_iteration": iteration,
        "lean_feedback": "none" if feedback is None else "compiler_diagnostics_and_static_policy",
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_completion_sha256": sha256(raw_completion.encode("utf-8")).hexdigest(),
        "seed": config.seed,
    }
    if feedback is not None:
        metadata["prior_candidate_id"] = feedback.candidate_id
        metadata["lean_feedback_sha256"] = feedback.sha256
    candidate = Candidate(
        candidate_id=_candidate_id(config.run_id, task.theorem_id, iteration),
        theorem_id=task.theorem_id,
        arm="llm_lean_repair",
        lean_code=lean_code,
        prompt_id=config.prompt_id,
        iteration=iteration,
        metadata=metadata,
    )
    result = evaluate_candidate(task, candidate, compiler)
    return LeanRepairAttempt(
        candidate=candidate,
        prompt=prompt,
        raw_completion=raw_completion,
        evaluation=result,
        feedback=feedback,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


def generate_verify_repair(
    tasks: Iterable[Task],
    provider: LeanRepairProvider,
    config: LeanRepairRunConfig,
    compiler: Compiler,
) -> list[LeanRepairAttempt]:
    """Run an initial attempt followed by bounded compiler-informed repairs.

    Every generated proof body is first passed through the static safety scan
    and then, unless rejected, the supplied independent Lean compiler. A failed
    result may be given to the provider only for the next bounded repair step.
    """
    task_list = list(tasks)
    assert_split_is_allowed(task_list, "development")
    attempts: list[LeanRepairAttempt] = []
    for task in task_list:
        initial_prompt = build_initial_prompt(task)
        initial_completion = provider.complete_initial(
            InitialGenerationRequest(
                theorem_id=task.theorem_id,
                prompt=initial_prompt,
                model_identifier=config.model_identifier,
                seed=config.seed,
            )
        )
        current = _capture_attempt(
            task=task,
            config=config,
            iteration=0,
            prompt=initial_prompt,
            completion=initial_completion,
            compiler=compiler,
            feedback=None,
        )
        attempts.append(current)

        for repair_iteration in range(1, config.max_repair_iterations + 1):
            if current.evaluation.compile_passed:
                break
            feedback = LeanCompilerFeedback.from_evaluation(current.candidate, current.evaluation)
            repair_prompt = build_repair_prompt(task, feedback, repair_iteration)
            repair_completion = provider.complete_repair(
                LeanRepairRequest(
                    theorem_id=task.theorem_id,
                    prompt=repair_prompt,
                    feedback=feedback,
                    model_identifier=config.model_identifier,
                    seed=config.seed,
                    repair_iteration=repair_iteration,
                )
            )
            current = _capture_attempt(
                task=task,
                config=config,
                iteration=repair_iteration,
                prompt=repair_prompt,
                completion=repair_completion,
                compiler=compiler,
                feedback=feedback,
            )
            attempts.append(current)
    return attempts


def calculate_lean_repair_metrics(
    attempts: Sequence[LeanRepairAttempt],
    *,
    max_repair_iterations: int = LEAN_REPAIR_POLICY["max_repair_iterations"],
) -> dict[str, Any]:
    """Calculate controlled initial and post-repair compile rates by theorem."""
    if not attempts:
        raise ValueError("cannot calculate metrics for an empty attempt set")
    if max_repair_iterations < 0:
        raise ValueError("max_repair_iterations must be non-negative")
    by_theorem: dict[str, list[LeanRepairAttempt]] = {}
    for attempt in attempts:
        by_theorem.setdefault(attempt.candidate.theorem_id, []).append(attempt)
    ordered = {
        theorem_id: sorted(items, key=lambda item: item.candidate.iteration)
        for theorem_id, items in by_theorem.items()
    }
    if any(items[0].candidate.iteration != 0 for items in ordered.values()):
        raise ValueError("each theorem must have an initial iteration 0 attempt")

    theorem_count = len(ordered)
    initial_passes = sum(items[0].evaluation.compile_passed for items in ordered.values())
    final_passes = sum(
        any(item.evaluation.compile_passed for item in items) for items in ordered.values()
    )
    repaired_successes = sum(
        not items[0].evaluation.compile_passed
        and any(item.evaluation.compile_passed for item in items[1:])
        for items in ordered.values()
    )
    exhausted = sum(
        not any(item.evaluation.compile_passed for item in items)
        and len(items) - 1 == max_repair_iterations
        for items in ordered.values()
    )
    repair_counts = [len(items) - 1 for items in ordered.values()]
    return {
        "theorem_count": theorem_count,
        "attempt_count": len(attempts),
        "compile_at_1": round(initial_passes / theorem_count, 6),
        "compile_after_lean_repair": round(final_passes / theorem_count, 6),
        "repaired_theorem_count": repaired_successes,
        "repair_budget_exhausted_theorem_count": exhausted,
        "mean_repair_iterations_executed": round(sum(repair_counts) / theorem_count, 6),
        "max_recorded_iteration": max(item.candidate.iteration for item in attempts),
    }


def write_lean_repair_attempt_records(path: Path, attempts: Sequence[LeanRepairAttempt]) -> None:
    """Persist each initial/repair attempt before publishing aggregate metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(attempt.to_record(), sort_keys=True) for attempt in attempts]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_lean_repair_evidence(
    path: Path,
    *,
    config: LeanRepairRunConfig,
    attempts: Sequence[LeanRepairAttempt],
    synthetic_demo: bool,
) -> None:
    """Write an auditable record of compiler-informed repair attempts."""
    payload = {
        "schema_version": 1,
        "synthetic_demo": synthetic_demo,
        "generation_policy": LEAN_REPAIR_POLICY,
        "pinned_toolchain": {
            "benchmark_commit": PINNED_MINIF2F_COMMIT,
            "lean_toolchain": PINNED_LEAN_TOOLCHAIN,
            "mathlib_revision": PINNED_MATHLIB_REVISION,
        },
        "run_config": asdict(config),
        "metrics": calculate_lean_repair_metrics(
            attempts,
            max_repair_iterations=config.max_repair_iterations,
        ),
        "attempts": [attempt.to_record() for attempt in attempts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
