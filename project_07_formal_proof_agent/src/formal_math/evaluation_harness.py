"""Reproducible evaluation primitives for Project 07.

This module deliberately separates candidate generation from proof evaluation.
Every study arm supplies the same candidate format; this harness performs the
independent static policy scan and post-hoc Lean compilation decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence


VALID_ARMS = frozenset({"llm_only", "llm_sympy", "llm_lean_repair"})
VALID_SPLITS = frozenset({"valid", "test"})
FORBIDDEN_SHORTCUTS = frozenset({"sorry", "admit", "axiom"})
_SHORTCUT_PATTERN = re.compile(r"\b(sorry|admit|axiom)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Task:
    """One theorem statement from a pinned benchmark or custom suite."""

    theorem_id: str
    split: str
    declaration: str
    imports: tuple[str, ...] = ()
    preamble: str = ""
    source: str = "miniF2F"

    def __post_init__(self) -> None:
        if not self.theorem_id:
            raise ValueError("theorem_id must not be empty")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if not self.declaration.strip():
            raise ValueError("declaration must not be empty")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Task":
        imports = record.get("imports", [])
        if not isinstance(imports, list) or not all(isinstance(item, str) for item in imports):
            raise ValueError("imports must be a list of strings")
        return cls(
            theorem_id=str(record["theorem_id"]),
            split=str(record["split"]),
            declaration=str(record["declaration"]),
            imports=tuple(imports),
            preamble=str(record.get("preamble", "")),
            source=str(record.get("source", "miniF2F")),
        )

    def render_context(self) -> str:
        """Render immutable task material visible to a proof generator.

        ``preamble`` permits a custom theorem suite to supply definitions while
        preserving the rule that the theorem declaration itself cannot be
        changed by a candidate.  Existing miniF2F tasks continue to use an
        empty preamble.
        """
        import_block = "\n".join(f"import {module}" for module in self.imports)
        blocks = [
            block
            for block in (import_block, self.preamble.strip(), self.declaration.strip())
            if block
        ]
        return "\n\n".join(blocks)

    def render_source(self, lean_code: str) -> str:
        """Render a standalone Lean candidate without allowing theorem edits."""
        blocks = [block for block in (self.render_context(), lean_code.strip()) if block]
        return "\n\n".join(blocks) + "\n"


@dataclass(frozen=True)
class Candidate:
    """Captured output of one generation or repair attempt."""

    candidate_id: str
    theorem_id: str
    arm: str
    lean_code: str
    prompt_id: str
    iteration: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.theorem_id or not self.prompt_id:
            raise ValueError("candidate_id, theorem_id, and prompt_id are required")
        if self.arm not in VALID_ARMS:
            raise ValueError(f"unsupported generation arm: {self.arm}")
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if not self.lean_code.strip():
            raise ValueError("lean_code must not be empty")

    @property
    def source_sha256(self) -> str:
        return sha256(self.lean_code.encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["source_sha256"] = self.source_sha256
        return record


@dataclass(frozen=True)
class CompilerOutcome:
    """Normalized output from the independent compiler invocation."""

    passed: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None


@dataclass(frozen=True)
class EvaluationResult:
    candidate_id: str
    theorem_id: str
    arm: str
    split: str
    status: str
    source_sha256: str
    static_rejections: tuple[str, ...]
    compiler_exit_code: int | None
    compiler_stdout: str
    compiler_stderr: str
    elapsed_seconds: float

    @property
    def compile_passed(self) -> bool:
        return self.status == "compiled"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["compile_passed"] = self.compile_passed
        return record


Compiler = Callable[[str], CompilerOutcome]


def static_rejection_reasons(lean_code: str) -> tuple[str, ...]:
    """Reject prohibited proof shortcuts before the compiler is contacted."""
    found = {match.group(1).lower() for match in _SHORTCUT_PATTERN.finditer(lean_code)}
    return tuple(sorted(found))


def assert_split_is_allowed(tasks: Iterable[Task], mode: str) -> None:
    """Prevent accidental test-split access during development.

    Development runs may contain only ``valid`` tasks.  A final evaluation run
    may contain only ``test`` tasks and must be selected explicitly by caller.
    """
    if mode not in {"development", "final_evaluation"}:
        raise ValueError("mode must be 'development' or 'final_evaluation'")
    allowed_split = "valid" if mode == "development" else "test"
    disallowed = sorted({task.split for task in tasks if task.split != allowed_split})
    if disallowed:
        raise PermissionError(
            f"{mode} permits only '{allowed_split}' tasks; found {', '.join(disallowed)}"
        )


def evaluate_candidate(task: Task, candidate: Candidate, compiler: Compiler) -> EvaluationResult:
    """Evaluate one candidate under the static policy then the compiler."""
    if task.theorem_id != candidate.theorem_id:
        raise ValueError("candidate theorem_id does not match task theorem_id")

    rejections = static_rejection_reasons(candidate.lean_code)
    if rejections:
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            theorem_id=task.theorem_id,
            arm=candidate.arm,
            split=task.split,
            status="rejected_static",
            source_sha256=candidate.source_sha256,
            static_rejections=rejections,
            compiler_exit_code=None,
            compiler_stdout="",
            compiler_stderr="compiler not invoked after static rejection",
            elapsed_seconds=0.0,
        )

    started = perf_counter()
    outcome = compiler(task.render_source(candidate.lean_code))
    elapsed = perf_counter() - started
    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        theorem_id=task.theorem_id,
        arm=candidate.arm,
        split=task.split,
        status="compiled" if outcome.passed else "failed_compile",
        source_sha256=candidate.source_sha256,
        static_rejections=(),
        compiler_exit_code=outcome.exit_code,
        compiler_stdout=outcome.stdout,
        compiler_stderr=outcome.stderr,
        elapsed_seconds=round(elapsed, 6),
    )


def evaluate_batch(
    tasks: Sequence[Task],
    candidates: Sequence[Candidate],
    compiler: Compiler,
    *,
    mode: str = "development",
) -> list[EvaluationResult]:
    """Evaluate a captured batch with split and theorem-identity controls."""
    assert_split_is_allowed(tasks, mode)
    task_by_id = {task.theorem_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("task theorem_id values must be unique")
    results: list[EvaluationResult] = []
    for candidate in candidates:
        try:
            task = task_by_id[candidate.theorem_id]
        except KeyError as error:
            raise ValueError(f"candidate references unknown theorem: {candidate.theorem_id}") from error
        results.append(evaluate_candidate(task, candidate, compiler))
    return results


def load_tasks_jsonl(path: Path) -> list[Task]:
    """Load an explicit task manifest; blank lines are ignored."""
    tasks: list[Task] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"task record on line {line_number} must be an object")
        tasks.append(Task.from_record(parsed))
    return tasks


def summarise_results(results: Sequence[EvaluationResult]) -> dict[str, int]:
    """Return transparent counts without estimating model performance."""
    summary = {"compiled": 0, "failed_compile": 0, "rejected_static": 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    summary["total"] = len(results)
    return summary


def write_evaluation_bundle(
    path: Path,
    *,
    tasks: Sequence[Task],
    candidates: Sequence[Candidate],
    results: Sequence[EvaluationResult],
    mode: str,
    synthetic_demo: bool = False,
) -> None:
    """Write an auditable result bundle containing candidates and decisions."""
    payload = {
        "schema_version": 1,
        "evaluation_mode": mode,
        "synthetic_demo": synthetic_demo,
        "task_count": len(tasks),
        "candidate_count": len(candidates),
        "summary": summarise_results(results),
        "tasks": [asdict(task) for task in tasks],
        "candidates": [candidate.to_record() for candidate in candidates],
        "results": [result.to_record() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class LeanSubprocessCompiler:
    """Compile a rendered candidate using a caller-supplied Lean command.

    Example command on Windows is ``[elan.exe, 'run', toolchain, 'lean']``.
    The caller supplies the miniF2F working directory when imports are needed.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not command:
            raise ValueError("Lean command must not be empty")
        self.command = tuple(command)
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds

    def __call__(self, source: str) -> CompilerOutcome:
        with tempfile.TemporaryDirectory(prefix="project07_candidate_") as directory:
            source_path = Path(directory) / "Candidate.lean"
            source_path.write_text(source, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [*self.command, str(source_path)],
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                return CompilerOutcome(
                    passed=False,
                    stdout=error.stdout or "",
                    stderr=(error.stderr or "") + f"\ncompiler timed out after {self.timeout_seconds} seconds",
                    exit_code=None,
                )
        return CompilerOutcome(
            passed=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
        )
