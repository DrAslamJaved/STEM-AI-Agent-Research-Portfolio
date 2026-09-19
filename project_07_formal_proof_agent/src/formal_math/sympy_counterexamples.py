"""SymPy-assisted diagnostics and exact counterexample search for Project 07.

This module deliberately keeps symbolic assistance separate from formal proof
validation.  SymPy may simplify a declared algebraic identity and search a
finite grid of exact rational witnesses, but it never decides whether a Lean
candidate is valid.  Candidates are captured first and are evaluated only by
the independent Phase 03 harness after generation has closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

try:  # Import lazily so the contract validator can run before dependency setup.
    import sympy as _sympy
except ModuleNotFoundError:  # pragma: no cover - exercised only without SymPy.
    _sympy = None

from formal_math.evaluation_harness import Candidate, EvaluationResult, Task, assert_split_is_allowed


SYMPY_WORKFLOW_POLICY = {
    "arm": "llm_sympy",
    "sympy_feedback_during_generation": "allowed_exact_diagnostics_and_counterexamples",
    "lean_feedback_during_generation": "forbidden",
    "evaluation_timing": "post_generation_only",
    "test_split_locked": True,
    "counterexample_arithmetic": "exact_rational_only",
}
_FENCED_BLOCK = re.compile(r"```(?:lean)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def _require_sympy() -> Any:
    """Return SymPy or explain how to install the declared project dependency."""
    if _sympy is None:
        raise RuntimeError(
            "SymPy is required for Phase 05. Install project_07_formal_proof_agent/"
            "requirements-phase05.txt before running the workflow."
        )
    return _sympy


@dataclass(frozen=True)
class SymbolicDiagnostic:
    """An auditable algebraic simplification result, not a formal proof."""

    claim_id: str
    lhs: str
    rhs: str
    residual: str
    equivalent_under_sympy_simplification: bool
    method: str = "sympy_simplify_exact_symbolic_residual"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["formal_validity_decision"] = "not_decided_by_sympy"
        return record


@dataclass(frozen=True)
class CounterexampleWitness:
    """A finite-grid, exact-rational witness against a candidate claim."""

    claim_id: str
    claim_description: str
    assignments: Mapping[str, str]
    observed_value: str
    examined_points: int
    search_method: str = "finite_exact_rational_grid"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["arithmetic"] = "exact_rational"
        record["formal_validity_decision"] = "not_decided_by_sympy"
        return record


@dataclass(frozen=True)
class SympyContext:
    """Only SymPy-produced context that may be supplied to the LLM arm."""

    diagnostics: tuple[SymbolicDiagnostic, ...] = ()
    counterexamples: tuple[CounterexampleWitness, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "diagnostics": [item.to_record() for item in self.diagnostics],
            "counterexamples": [item.to_record() for item in self.counterexamples],
            "formal_validity_decision": "reserved_for_independent_lean_evaluation",
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_record(), sort_keys=True).encode("utf-8")
        return sha256(encoded).hexdigest()

    def render_for_prompt(self) -> str:
        """Render a compact, provenance-preserving SymPy-only prompt block."""
        lines = [
            "SymPy-only diagnostic context (not Lean compiler feedback):",
            "- SymPy can guide algebra and exact witness search; it cannot establish Lean validity.",
        ]
        for diagnostic in self.diagnostics:
            verdict = "residual is zero" if diagnostic.equivalent_under_sympy_simplification else "residual is nonzero"
            lines.append(
                f"- {diagnostic.claim_id}: {diagnostic.lhs} = {diagnostic.rhs}; "
                f"residual {diagnostic.residual} ({verdict})."
            )
        for witness in self.counterexamples:
            assignments = ", ".join(f"{name}={value}" for name, value in witness.assignments.items())
            lines.append(
                f"- Exact counterexample to {witness.claim_id}: {assignments}; "
                f"observed value {witness.observed_value}."
            )
        return "\n".join(lines)


def diagnose_symbolic_equality(
    claim_id: str,
    lhs: str,
    rhs: str,
    *,
    variables: Sequence[str],
) -> SymbolicDiagnostic:
    """Simplify an algebraic residual using exact symbolic expressions.

    A zero residual is a useful diagnostic for the supplied expressions.  It is
    explicitly not recorded as a formal Lean proof.
    """
    sympy = _require_sympy()
    if not claim_id or not lhs or not rhs:
        raise ValueError("claim_id, lhs, and rhs are required")
    if not variables or any(not name.isidentifier() for name in variables):
        raise ValueError("variables must be non-empty valid identifiers")
    symbols = {name: sympy.symbols(name) for name in variables}
    left_expression = sympy.sympify(lhs, locals=symbols)
    right_expression = sympy.sympify(rhs, locals=symbols)
    residual = sympy.simplify(left_expression - right_expression)
    return SymbolicDiagnostic(
        claim_id=claim_id,
        lhs=str(left_expression),
        rhs=str(right_expression),
        residual=str(residual),
        equivalent_under_sympy_simplification=bool(residual == 0),
    )


def exact_rational_grid(numerators: Sequence[int], denominator: int) -> tuple[Any, ...]:
    """Create a de-duplicated exact grid; floating-point values are never used."""
    sympy = _require_sympy()
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if not numerators:
        raise ValueError("at least one numerator is required")
    values = tuple(sympy.Rational(value, denominator) for value in numerators)
    if len(set(values)) != len(values):
        raise ValueError("the exact grid must not contain duplicate values")
    return values


def singleton_jaccard(left: Any, right: Any) -> Any:
    """Return exact singleton fuzzy-set Jaccard similarity with J(0, 0) = 1."""
    sympy = _require_sympy()
    left_value = sympy.Rational(left)
    right_value = sympy.Rational(right)
    if not (0 <= left_value <= 1 and 0 <= right_value <= 1):
        raise ValueError("fuzzy memberships must lie in the closed interval [0, 1]")
    if left_value == 0 and right_value == 0:
        return sympy.Integer(1)
    return sympy.simplify(sympy.Min(left_value, right_value) / sympy.Max(left_value, right_value))


def find_exact_counterexample(
    *,
    claim_id: str,
    claim_description: str,
    variables: Sequence[str],
    grid: Sequence[Any],
    claim_holds: Callable[[Mapping[str, Any]], bool],
    observed_value: Callable[[Mapping[str, Any]], Any],
    max_evaluations: int = 10_000,
) -> CounterexampleWitness | None:
    """Search a finite exact-rational grid for one falsifying assignment.

    ``claim_holds`` must return a Python boolean.  The function intentionally
    explores only the declared finite domain, so a missing witness is not a
    proof that the claim holds outside that domain.
    """
    sympy = _require_sympy()
    if not claim_id or not claim_description:
        raise ValueError("claim_id and claim_description are required")
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("variables must be a non-empty unique sequence")
    if any(not name.isidentifier() for name in variables):
        raise ValueError("variables must contain valid identifiers")
    if not grid:
        raise ValueError("grid must not be empty")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")

    exact_grid = tuple(sympy.Rational(value) for value in grid)
    if any(not value.is_Rational for value in exact_grid):
        raise ValueError("counterexample search accepts exact rational values only")

    examined = 0
    for values in product(exact_grid, repeat=len(variables)):
        examined += 1
        if examined > max_evaluations:
            raise ValueError("counterexample search exceeded max_evaluations")
        assignment = dict(zip(variables, values, strict=True))
        if bool(claim_holds(assignment)):
            continue
        result = sympy.simplify(observed_value(assignment))
        return CounterexampleWitness(
            claim_id=claim_id,
            claim_description=claim_description,
            assignments={name: str(value) for name, value in assignment.items()},
            observed_value=str(result),
            examined_points=examined,
        )
    return None


def find_singleton_jaccard_identity_counterexample() -> CounterexampleWitness:
    """Refute the false claim that unequal nonzero singleton sets always score one."""
    grid = exact_rational_grid((1, 2, 3, 4), 4)
    witness = find_exact_counterexample(
        claim_id="singleton_jaccard_nonzero_identity",
        claim_description=(
            "For all nonzero singleton fuzzy memberships a and b, J(a, b) = 1."
        ),
        variables=("a", "b"),
        grid=grid,
        claim_holds=lambda values: singleton_jaccard(values["a"], values["b"]) == 1,
        observed_value=lambda values: singleton_jaccard(values["a"], values["b"]),
    )
    if witness is None:  # Defensive: the fixed grid contains unequal values.
        raise AssertionError("expected an exact counterexample on the declared grid")
    return witness


@dataclass(frozen=True)
class SympyRunConfig:
    """Pre-registered limits and provenance for an LLM+SymPy run."""

    run_id: str
    model_identifier: str
    prompt_id: str
    max_sympy_repair_iterations: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.model_identifier or not self.prompt_id:
            raise ValueError("run_id, model_identifier, and prompt_id are required")
        if self.max_sympy_repair_iterations < 0:
            raise ValueError("max_sympy_repair_iterations must be non-negative")


@dataclass(frozen=True)
class SympyGenerationRequest:
    """Provider input: frozen task plus SymPy context, never Lean diagnostics."""

    theorem_id: str
    prompt: str
    sympy_context: str
    model_identifier: str
    seed: int | None
    iteration: int


@dataclass(frozen=True)
class SympyCompletion:
    """A raw completion retained before independent post-hoc evaluation."""

    text: str
    model_identifier: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class SympyCompletionProvider(Protocol):
    """Provider interface that is intentionally incapable of receiving Lean output."""

    def complete(self, request: SympyGenerationRequest) -> SympyCompletion:
        ...


@dataclass(frozen=True)
class SympyGeneratedCandidate:
    """Candidate and all pre-evaluation SymPy provenance needed to audit it."""

    candidate: Candidate
    prompt: str
    raw_completion: str
    context: SympyContext
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
                "sympy_context": self.context.to_record(),
                "sympy_context_sha256": self.context.sha256,
            }
        )
        return record


def build_llm_sympy_prompt(task: Task, context: SympyContext, iteration: int) -> str:
    """Create a prompt that exposes only frozen task data and SymPy evidence."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    imports = "\n".join(f"import {module}" for module in task.imports)
    frozen_task = "\n\n".join(part for part in (imports, task.declaration.strip()) if part)
    return (
        "Produce only the Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "You may use only the SymPy context below; you will receive no Lean compiler "
        "diagnostics, compilation status, search results, or prior candidate text.\n"
        "SymPy evidence is advisory and cannot establish formal validity.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{context.render_for_prompt()}\n\n"
        f"Frozen theorem (SymPy repair iteration {iteration}):\n{frozen_task}\n"
    )


def normalise_proof_body(raw_completion: str) -> str:
    """Remove a Markdown fence while retaining only the returned Lean proof body."""
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


def _candidate_id(run_id: str, theorem_id: str, iteration: int) -> str:
    material = f"{run_id}|{theorem_id}|{iteration}".encode("utf-8")
    return f"sympy-{sha256(material).hexdigest()[:16]}"


def generate_llm_sympy_candidates(
    tasks: Iterable[Task],
    provider: SympyCompletionProvider,
    config: SympyRunConfig,
    contexts_by_theorem: Mapping[str, SympyContext],
) -> list[SympyGeneratedCandidate]:
    """Generate a closed LLM+SymPy batch without invoking an evaluator."""
    task_list = list(tasks)
    assert_split_is_allowed(task_list, "development")
    generated: list[SympyGeneratedCandidate] = []
    for task in task_list:
        try:
            context = contexts_by_theorem[task.theorem_id]
        except KeyError as error:
            raise ValueError(f"missing SymPy context for {task.theorem_id}") from error
        for iteration in range(config.max_sympy_repair_iterations + 1):
            prompt = build_llm_sympy_prompt(task, context, iteration)
            request = SympyGenerationRequest(
                theorem_id=task.theorem_id,
                prompt=prompt,
                sympy_context=context.render_for_prompt(),
                model_identifier=config.model_identifier,
                seed=config.seed,
                iteration=iteration,
            )
            completion = provider.complete(request)
            raw_completion = completion.text
            lean_code = normalise_proof_body(raw_completion)
            model_identifier = completion.model_identifier or config.model_identifier
            metadata = {
                "generation_policy": "sympy_only_no_lean_feedback",
                "sympy_feedback": "exact_diagnostics_and_counterexamples",
                "lean_feedback": "forbidden",
                "evaluation_feedback": "forbidden",
                "model_identifier": model_identifier,
                "iteration": iteration,
                "seed": config.seed,
                "sympy_context_sha256": context.sha256,
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_completion_sha256": sha256(raw_completion.encode("utf-8")).hexdigest(),
            }
            candidate = Candidate(
                candidate_id=_candidate_id(config.run_id, task.theorem_id, iteration),
                theorem_id=task.theorem_id,
                arm="llm_sympy",
                lean_code=lean_code,
                prompt_id=config.prompt_id,
                iteration=iteration,
                metadata=metadata,
            )
            generated.append(
                SympyGeneratedCandidate(
                    candidate=candidate,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    context=context,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                )
            )
    return generated


def write_sympy_generation_records(path: Path, generated: Sequence[SympyGeneratedCandidate]) -> None:
    """Persist the closed candidate batch before post-hoc Lean evaluation begins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.to_record(), sort_keys=True) for item in generated]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def calculate_sympy_metrics(
    generated: Sequence[SympyGeneratedCandidate],
    results: Sequence[EvaluationResult],
) -> dict[str, Any]:
    """Calculate initial and bounded-repair compilation rates from post-hoc results."""
    if not generated:
        raise ValueError("cannot calculate metrics for an empty candidate batch")
    result_by_id = {result.candidate_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("result candidate_id values must be unique")

    by_theorem: dict[str, list[SympyGeneratedCandidate]] = {}
    for item in generated:
        candidate_id = item.candidate.candidate_id
        if candidate_id not in result_by_id:
            raise ValueError(f"missing evaluation result for {candidate_id}")
        by_theorem.setdefault(item.candidate.theorem_id, []).append(item)
    ordered = {
        theorem_id: sorted(items, key=lambda item: item.candidate.iteration)
        for theorem_id, items in by_theorem.items()
    }
    if any(items[0].candidate.iteration != 0 for items in ordered.values()):
        raise ValueError("each theorem must retain an initial iteration 0 candidate")

    theorem_count = len(ordered)
    initial_passes = sum(
        result_by_id[items[0].candidate.candidate_id].compile_passed for items in ordered.values()
    )
    any_passes = sum(
        any(result_by_id[item.candidate.candidate_id].compile_passed for item in items)
        for items in ordered.values()
    )
    return {
        "theorem_count": theorem_count,
        "candidate_count": len(generated),
        "compile_at_1": round(initial_passes / theorem_count, 6),
        "compile_after_sympy_repair": round(any_passes / theorem_count, 6),
        "max_recorded_iteration": max(item.candidate.iteration for item in generated),
    }


def write_sympy_workflow_evidence(
    path: Path,
    *,
    config: SympyRunConfig,
    generated: Sequence[SympyGeneratedCandidate],
    results: Sequence[EvaluationResult],
    synthetic_demo: bool,
) -> None:
    """Write an auditable evidence bundle after independent evaluation finishes."""
    contexts = {item.candidate.theorem_id: item.context for item in generated}
    payload = {
        "schema_version": 1,
        "synthetic_demo": synthetic_demo,
        "generation_completed_before_evaluation": True,
        "generation_policy": SYMPY_WORKFLOW_POLICY,
        "run_config": asdict(config),
        "metrics": calculate_sympy_metrics(generated, results),
        "sympy_context_by_theorem": {
            theorem_id: context.to_record() for theorem_id, context in sorted(contexts.items())
        },
        "generated_candidates": [item.to_record() for item in generated],
        "evaluation_results": [result.to_record() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
