"""Controlled, tool-free candidate generation for Project 07 Phase 04.

The baseline intentionally knows nothing about Lean diagnostics, SymPy output,
counterexamples, or prior attempts. It generates captured proof bodies first;
Phase 03 evaluates them only after the generation batch is closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Protocol, Sequence

from formal_math.evaluation_harness import Candidate, EvaluationResult, Task, assert_split_is_allowed


LLM_ONLY_POLICY = {
    "arm": "llm_only",
    "verification_feedback_during_generation": "forbidden",
    "repair_iterations": 0,
    "tool_calls_during_generation": "forbidden",
    "evaluation_timing": "post_generation_only",
}
_FENCED_BLOCK = re.compile(r"```(?:lean)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class BaselineRunConfig:
    """Immutable metadata for one pre-registered baseline generation batch."""

    run_id: str
    model_identifier: str
    prompt_id: str
    samples_per_theorem: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.model_identifier or not self.prompt_id:
            raise ValueError("run_id, model_identifier, and prompt_id are required")
        if self.samples_per_theorem < 1:
            raise ValueError("samples_per_theorem must be at least one")


@dataclass(frozen=True)
class GenerationRequest:
    """The only data supplied to the baseline completion provider."""

    theorem_id: str
    prompt: str
    model_identifier: str
    seed: int | None
    sample_index: int


@dataclass(frozen=True)
class Completion:
    """One raw model response captured before independent evaluation."""

    text: str
    model_identifier: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class CompletionProvider(Protocol):
    """A provider that receives generation input only, never evaluation output."""

    def complete(self, request: GenerationRequest) -> Completion:
        ...


@dataclass(frozen=True)
class GeneratedCandidate:
    """Candidate provenance retained before post-hoc Lean evaluation."""

    candidate: Candidate
    prompt: str
    raw_completion: str
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
            }
        )
        return record


def build_llm_only_prompt(task: Task) -> str:
    """Return a theorem-only prompt with no verification or repair context."""
    imports = "\n".join(f"import {module}" for module in task.imports)
    context = "\n\n".join(part for part in (imports, task.declaration.strip()) if part)
    return (
        "Produce only the Lean proof body for the frozen theorem below.\n"
        "Do not restate, edit, or weaken the theorem declaration.\n"
        "You have no tools and will receive no feedback or repair suggestions.\n"
        "Do not use sorry, admit, or axiom.\n\n"
        f"{context}\n"
    )


def normalise_proof_body(raw_completion: str) -> str:
    """Remove a Markdown fence while preserving the generated proof."""
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


def _candidate_id(run_id: str, theorem_id: str, sample_index: int) -> str:
    material = f"{run_id}|{theorem_id}|{sample_index}".encode("utf-8")
    return f"baseline-{sha256(material).hexdigest()[:16]}"


def generate_llm_only_candidates(
    tasks: Iterable[Task],
    provider: CompletionProvider,
    config: BaselineRunConfig,
) -> list[GeneratedCandidate]:
    """Generate candidates without passing verification feedback to provider."""
    task_list = list(tasks)
    assert_split_is_allowed(task_list, "development")
    generated: list[GeneratedCandidate] = []
    for task in task_list:
        prompt = build_llm_only_prompt(task)
        for sample_index in range(config.samples_per_theorem):
            request = GenerationRequest(
                theorem_id=task.theorem_id,
                prompt=prompt,
                model_identifier=config.model_identifier,
                seed=config.seed,
                sample_index=sample_index,
            )
            completion = provider.complete(request)
            raw_completion = completion.text
            lean_code = normalise_proof_body(raw_completion)
            model_identifier = completion.model_identifier or config.model_identifier
            metadata = {
                "baseline_arm": "llm_only",
                "generation_feedback": "none",
                "generation_policy": "tool_free_no_feedback",
                "model_identifier": model_identifier,
                "sample_index": sample_index,
                "seed": config.seed,
                "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_completion_sha256": sha256(raw_completion.encode("utf-8")).hexdigest(),
            }
            candidate = Candidate(
                candidate_id=_candidate_id(config.run_id, task.theorem_id, sample_index),
                theorem_id=task.theorem_id,
                arm="llm_only",
                lean_code=lean_code,
                prompt_id=config.prompt_id,
                iteration=0,
                metadata=metadata,
            )
            generated.append(
                GeneratedCandidate(
                    candidate=candidate,
                    prompt=prompt,
                    raw_completion=raw_completion,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                )
            )
    return generated


def write_generation_records(path: Path, generated: Sequence[GeneratedCandidate]) -> None:
    """Persist immutable baseline-generation records before evaluation begins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.to_record(), sort_keys=True) for item in generated]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def calculate_baseline_metrics(
    generated: Sequence[GeneratedCandidate],
    results: Sequence[EvaluationResult],
    *,
    ks: Sequence[int] = (1,),
) -> dict[str, Any]:
    """Compute observed compile@1 and pass@k from post-hoc results."""
    if not generated:
        raise ValueError("cannot calculate metrics for an empty candidate batch")
    result_by_id = {result.candidate_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("result candidate_id values must be unique")

    by_theorem: dict[str, list[GeneratedCandidate]] = {}
    for item in generated:
        if item.candidate.candidate_id not in result_by_id:
            raise ValueError(f"missing evaluation result for {item.candidate.candidate_id}")
        by_theorem.setdefault(item.candidate.theorem_id, []).append(item)

    ordered = {
        theorem_id: sorted(
            items,
            key=lambda item: int(item.candidate.metadata["sample_index"]),
        )
        for theorem_id, items in by_theorem.items()
    }
    theorem_count = len(ordered)
    metrics: dict[str, Any] = {
        "theorem_count": theorem_count,
        "candidate_count": len(generated),
        "compile_at_1": round(
            sum(result_by_id[items[0].candidate.candidate_id].compile_passed for items in ordered.values())
            / theorem_count,
            6,
        ),
        "observed_pass_at_k": {},
    }
    for k in ks:
        if k < 1:
            raise ValueError("k values must be positive")
        if any(len(items) < k for items in ordered.values()):
            raise ValueError(f"at least {k} generated candidates per theorem are required")
        successes = sum(
            any(result_by_id[item.candidate.candidate_id].compile_passed for item in items[:k])
            for items in ordered.values()
        )
        metrics["observed_pass_at_k"][str(k)] = round(successes / theorem_count, 6)
    return metrics


def write_baseline_evidence(
    path: Path,
    *,
    config: BaselineRunConfig,
    generated: Sequence[GeneratedCandidate],
    results: Sequence[EvaluationResult],
    synthetic_demo: bool,
) -> None:
    """Write an auditable Phase 04 run bundle after independent evaluation."""
    metrics = calculate_baseline_metrics(
        generated,
        results,
        ks=tuple(range(1, config.samples_per_theorem + 1)),
    )
    payload = {
        "schema_version": 1,
        "synthetic_demo": synthetic_demo,
        "generation_completed_before_evaluation": True,
        "generation_policy": LLM_ONLY_POLICY,
        "run_config": asdict(config),
        "metrics": metrics,
        "generated_candidates": [item.to_record() for item in generated],
        "evaluation_results": [result.to_record() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
