"""Locked final-evaluation and release-gate utilities for Project 07.

The module aggregates captured candidate evaluations; it does not generate
proofs, call an LLM, or treat SymPy output as formal verification. Every
reported proof-success metric comes from the independent Lean evaluation status
already recorded for that candidate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from formal_math.evaluation_harness import VALID_ARMS


EXPECTED_ARMS = ("llm_only", "llm_sympy", "llm_lean_repair")
FORMAL_VALIDITY_DECISION = "independent_lean_compilation"
VALID_STATUSES = frozenset({"compiled", "failed_compile", "rejected_static"})
VALID_KINDS = frozenset({"proof", "refutation"})
ARM_POLICY = {
    "llm_only": {
        "lean_feedback": "forbidden",
        "sympy_feedback": "forbidden",
        "max_repair_iterations": 0,
    },
    "llm_sympy": {
        "lean_feedback": "forbidden",
        "sympy_feedback": "allowed",
        "max_repair_iterations": 0,
    },
    "llm_lean_repair": {
        "lean_feedback": "allowed",
        "sympy_feedback": "not_required",
        "max_repair_iterations": 2,
    },
}


@dataclass(frozen=True)
class FinalAttempt:
    """One immutable candidate-evaluation record for the locked test split."""

    candidate_id: str
    theorem_id: str
    arm: str
    split: str
    sample_index: int
    iteration: int
    status: str
    kind: str = "proof"
    static_rejections: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    human_intervention: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.theorem_id:
            raise ValueError("candidate_id and theorem_id are required")
        if self.arm not in EXPECTED_ARMS:
            raise ValueError(f"unsupported arm: {self.arm}")
        if self.arm not in VALID_ARMS:
            raise ValueError(f"arm is not supported by the Phase 03 harness: {self.arm}")
        if self.split not in {"valid", "test"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.sample_index < 1:
            raise ValueError("sample_index must be at least one")
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unsupported evaluation status: {self.status}")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unsupported task kind: {self.kind}")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be non-negative")
        if not isinstance(self.human_intervention, bool):
            raise ValueError("human_intervention must be a boolean")

    @property
    def compiled(self) -> bool:
        return self.status == "compiled"

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FinalAttempt":
        """Normalize Phase 03/04/05/06-style captured candidate records."""

        candidate = record.get("candidate", record)
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate record must be a mapping")
        evaluation = record.get("evaluation", record.get("evaluation_result", record))
        if not isinstance(evaluation, Mapping):
            raise ValueError("evaluation record must be a mapping")

        static_rejections = evaluation.get(
            "static_rejections",
            record.get("static_rejections", ()),
        )
        if not isinstance(static_rejections, (list, tuple)) or not all(
            isinstance(item, str) for item in static_rejections
        ):
            raise ValueError("static_rejections must be a list or tuple of strings")

        return cls(
            candidate_id=str(candidate.get("candidate_id", record.get("candidate_id", ""))),
            theorem_id=str(candidate.get("theorem_id", record.get("theorem_id", ""))),
            arm=str(candidate.get("arm", evaluation.get("arm", record.get("arm", "")))),
            split=str(evaluation.get("split", candidate.get("split", record.get("split", "")))),
            sample_index=int(record.get("sample_index", candidate.get("sample_index", 1))),
            iteration=int(record.get("iteration", candidate.get("iteration", 0))),
            status=str(evaluation.get("status", record.get("status", ""))),
            kind=str(record.get("kind", candidate.get("kind", "proof"))),
            static_rejections=tuple(static_rejections),
            elapsed_seconds=float(
                evaluation.get("elapsed_seconds", record.get("elapsed_seconds", 0.0))
            ),
            input_tokens=int(record.get("input_tokens", candidate.get("input_tokens", 0))),
            output_tokens=int(record.get("output_tokens", candidate.get("output_tokens", 0))),
            estimated_cost_usd=float(record.get("estimated_cost_usd", 0.0)),
            human_intervention=bool(record.get("human_intervention", False)),
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        # JSON has no tuple type.  Normalize before both hashing and comparison
        # so a persisted release bundle can be reconstructed byte-for-byte.
        record["static_rejections"] = list(self.static_rejections)
        return record


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    value = _require_nonempty_string(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def validate_release_manifest(manifest: Mapping[str, Any]) -> None:
    """Check preregistration and provenance controls before aggregation."""

    if manifest.get("schema_version") != 1:
        raise ValueError("release manifest schema_version must be 1")
    if manifest.get("phase") != "08":
        raise ValueError("release manifest phase must be '08'")
    if manifest.get("evaluation_split") != "test":
        raise ValueError("Phase 08 accepts only the locked test split")
    if manifest.get("test_split_locked") is not True:
        raise ValueError("test_split_locked must be true")
    if manifest.get("formal_validity_decision") != FORMAL_VALIDITY_DECISION:
        raise ValueError("formal validity must remain independent Lean compilation")

    samples_per_theorem = manifest.get("samples_per_theorem")
    if not isinstance(samples_per_theorem, int) or samples_per_theorem < 1:
        raise ValueError("samples_per_theorem must be a positive integer")

    max_repair_iterations = manifest.get("max_repair_iterations")
    if max_repair_iterations != ARM_POLICY["llm_lean_repair"]["max_repair_iterations"]:
        raise ValueError("max_repair_iterations must match the registered Lean repair budget")

    arms = _require_mapping(manifest.get("arms"), "arms")
    if set(arms) != set(EXPECTED_ARMS):
        raise ValueError("release manifest must register exactly the three study arms")
    for arm, required_policy in ARM_POLICY.items():
        observed = _require_mapping(arms.get(arm), f"arms.{arm}")
        for key, expected in required_policy.items():
            if observed.get(key) != expected:
                raise ValueError(f"arms.{arm}.{key} must be {expected!r}")

    _require_nonempty_string(manifest.get("run_id"), "run_id")
    _require_nonempty_string(manifest.get("model_identifier"), "model_identifier")
    if "seed" not in manifest:
        raise ValueError("release manifest must record a seed")
    if not isinstance(manifest.get("synthetic_demo"), bool):
        raise ValueError("synthetic_demo must be a boolean")

    provenance = _require_mapping(manifest.get("provenance"), "provenance")
    for key in (
        "benchmark_commit",
        "lean_toolchain",
        "mathlib_revision",
        "task_manifest_sha256",
        "candidate_records_sha256",
    ):
        if key.endswith("_sha256"):
            _require_sha256(provenance.get(key), f"provenance.{key}")
        else:
            _require_nonempty_string(provenance.get(key), f"provenance.{key}")

    human_review = _require_mapping(manifest.get("human_review"), "human_review")
    if not isinstance(human_review.get("approved"), bool):
        raise ValueError("human_review.approved must be a boolean")


def validate_final_attempts(
    attempts: Sequence[FinalAttempt],
    manifest: Mapping[str, Any],
) -> tuple[FinalAttempt, ...]:
    """Reject incomplete, non-comparable, or development-split evidence."""

    validate_release_manifest(manifest)
    if not attempts:
        raise ValueError("final evaluation requires at least one attempt")

    samples_per_theorem = int(manifest["samples_per_theorem"])
    max_repair_iterations = int(manifest["max_repair_iterations"])
    theorem_sets: dict[str, set[str]] = {arm: set() for arm in EXPECTED_ARMS}
    initial_samples: dict[tuple[str, str], set[int]] = defaultdict(set)
    theorem_kinds: dict[str, str] = {}
    seen: set[tuple[str, str, int, int]] = set()

    for attempt in attempts:
        if attempt.split != "test":
            raise PermissionError("Phase 08 may aggregate only locked test-split attempts")
        if attempt.sample_index > samples_per_theorem:
            raise ValueError("attempt sample_index exceeds samples_per_theorem")
        if attempt.arm == "llm_lean_repair":
            if attempt.iteration > max_repair_iterations:
                raise ValueError("Lean repair iteration exceeds the registered budget")
        elif attempt.iteration != 0:
            raise ValueError(f"{attempt.arm} must not contain repair iterations")

        key = (attempt.arm, attempt.theorem_id, attempt.sample_index, attempt.iteration)
        if key in seen:
            raise ValueError("duplicate arm/theorem/sample/iteration record")
        seen.add(key)
        theorem_sets[attempt.arm].add(attempt.theorem_id)
        if attempt.iteration == 0:
            initial_samples[(attempt.arm, attempt.theorem_id)].add(attempt.sample_index)

        existing_kind = theorem_kinds.setdefault(attempt.theorem_id, attempt.kind)
        if existing_kind != attempt.kind:
            raise ValueError("theorem kind must remain identical across every arm")

    canonical_theorems = theorem_sets["llm_only"]
    if not canonical_theorems:
        raise ValueError("llm_only must contain at least one theorem")
    for arm, theorem_ids in theorem_sets.items():
        if theorem_ids != canonical_theorems:
            raise ValueError(f"{arm} does not evaluate the identical theorem set")

    expected_samples = set(range(1, samples_per_theorem + 1))
    for arm in EXPECTED_ARMS:
        for theorem_id in canonical_theorems:
            observed = initial_samples[(arm, theorem_id)]
            if observed != expected_samples:
                raise ValueError(
                    f"{arm}/{theorem_id} initial samples must be exactly {sorted(expected_samples)}"
                )

    return tuple(attempts)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _group_attempts(attempts: Iterable[FinalAttempt]) -> dict[str, dict[str, list[FinalAttempt]]]:
    grouped: dict[str, dict[str, list[FinalAttempt]]] = {
        arm: defaultdict(list) for arm in EXPECTED_ARMS
    }
    for attempt in attempts:
        grouped[attempt.arm][attempt.theorem_id].append(attempt)
    for arm in grouped:
        for theorem_id in grouped[arm]:
            grouped[arm][theorem_id].sort(key=lambda item: (item.sample_index, item.iteration))
    return grouped


def _theorem_compiles_at_1(items: Sequence[FinalAttempt]) -> bool:
    return any(item.compiled for item in items if item.sample_index == 1 and item.iteration == 0)


def _theorem_passes_at_k(items: Sequence[FinalAttempt], k: int) -> bool:
    return any(item.compiled for item in items if item.iteration == 0 and item.sample_index <= k)


def _theorem_compiles_after_workflow(items: Sequence[FinalAttempt]) -> bool:
    return any(item.compiled for item in items)


def calculate_arm_metrics(
    attempts: Sequence[FinalAttempt],
    *,
    samples_per_theorem: int,
) -> dict[str, dict[str, Any]]:
    """Report transparent compile and refutation counts for each study arm."""

    grouped = _group_attempts(attempts)
    metrics: dict[str, dict[str, Any]] = {}
    for arm in EXPECTED_ARMS:
        by_theorem = grouped[arm]
        theorem_ids = sorted(by_theorem)
        theorem_count = len(theorem_ids)
        refutation_ids = [
            theorem_id for theorem_id in theorem_ids if by_theorem[theorem_id][0].kind == "refutation"
        ]
        static_rejections = sum(
            len(item.static_rejections) for theorem_id in theorem_ids for item in by_theorem[theorem_id]
        )
        arm_attempts = [
            item for theorem_id in theorem_ids for item in by_theorem[theorem_id]
        ]
        metrics[arm] = {
            "theorem_count": theorem_count,
            "attempt_count": len(arm_attempts),
            "compile_at_1": _rate(
                sum(_theorem_compiles_at_1(by_theorem[theorem_id]) for theorem_id in theorem_ids),
                theorem_count,
            ),
            "pass_at_k": _rate(
                sum(
                    _theorem_passes_at_k(by_theorem[theorem_id], samples_per_theorem)
                    for theorem_id in theorem_ids
                ),
                theorem_count,
            ),
            "compile_after_allowed_workflow": _rate(
                sum(
                    _theorem_compiles_after_workflow(by_theorem[theorem_id])
                    for theorem_id in theorem_ids
                ),
                theorem_count,
            ),
            "refutation_theorem_count": len(refutation_ids),
            "refutation_success_rate": _rate(
                sum(
                    _theorem_compiles_after_workflow(by_theorem[theorem_id])
                    for theorem_id in refutation_ids
                ),
                len(refutation_ids),
            ),
            "static_rejection_count": static_rejections,
            "total_elapsed_seconds": round(sum(item.elapsed_seconds for item in arm_attempts), 6),
            "mean_elapsed_seconds_per_attempt": round(
                sum(item.elapsed_seconds for item in arm_attempts) / len(arm_attempts), 6
            ) if arm_attempts else 0.0,
            "input_token_count": sum(item.input_tokens for item in arm_attempts),
            "output_token_count": sum(item.output_tokens for item in arm_attempts),
            "estimated_cost_usd": round(
                sum(item.estimated_cost_usd for item in arm_attempts), 6
            ),
            "human_intervention_attempt_count": sum(
                item.human_intervention for item in arm_attempts
            ),
        }
    return metrics


def calculate_paired_comparisons(attempts: Sequence[FinalAttempt]) -> dict[str, dict[str, Any]]:
    """Compare each tool-assisted arm with LLM-only by theorem identity."""

    grouped = _group_attempts(attempts)
    baseline = grouped["llm_only"]
    comparisons: dict[str, dict[str, Any]] = {}
    for arm in ("llm_sympy", "llm_lean_repair"):
        paired = {
            "both_compiled": 0,
            "llm_only_only": 0,
            f"{arm}_only": 0,
            "both_failed": 0,
        }
        for theorem_id in sorted(baseline):
            baseline_pass = _theorem_compiles_after_workflow(baseline[theorem_id])
            arm_pass = _theorem_compiles_after_workflow(grouped[arm][theorem_id])
            if baseline_pass and arm_pass:
                paired["both_compiled"] += 1
            elif baseline_pass:
                paired["llm_only_only"] += 1
            elif arm_pass:
                paired[f"{arm}_only"] += 1
            else:
                paired["both_failed"] += 1
        theorem_count = len(baseline)
        discordant = paired["llm_only_only"] + paired[f"{arm}_only"]
        smaller_discordant_count = min(paired["llm_only_only"], paired[f"{arm}_only"])
        exact_two_sided_p_value = (
            min(
                1.0,
                2.0
                * sum(comb(discordant, index) for index in range(smaller_discordant_count + 1))
                / (2**discordant),
            )
            if discordant
            else 1.0
        )
        arm_rate = _rate(
            sum(
                _theorem_compiles_after_workflow(grouped[arm][theorem_id])
                for theorem_id in baseline
            ),
            theorem_count,
        )
        baseline_rate = _rate(
            sum(
                _theorem_compiles_after_workflow(baseline[theorem_id])
                for theorem_id in baseline
            ),
            theorem_count,
        )
        comparisons[arm] = {
            "theorem_count": theorem_count,
            "compile_after_allowed_workflow_delta_vs_llm_only": round(arm_rate - baseline_rate, 6),
            "paired_outcomes": paired,
            "discordant_pair_count": discordant,
            "exact_mcnemar_two_sided_p_value": round(exact_two_sided_p_value, 6),
        }
    return comparisons


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_release_bundle(
    manifest: Mapping[str, Any],
    attempts: Sequence[FinalAttempt],
) -> dict[str, Any]:
    """Create an auditable evaluation bundle and conservative release decision."""

    checked_attempts = validate_final_attempts(attempts, manifest)
    arm_metrics = calculate_arm_metrics(
        checked_attempts,
        samples_per_theorem=int(manifest["samples_per_theorem"]),
    )
    synthetic_demo = bool(manifest["synthetic_demo"])
    human_approved = bool(manifest["human_review"]["approved"])
    blockers: list[str] = []
    if synthetic_demo:
        blockers.append("synthetic_demo=true; this fixture cannot establish a final research result")
    if not human_approved:
        blockers.append("human_review.approved is false")

    return {
        "schema_version": 1,
        "phase": "08",
        "run_id": manifest["run_id"],
        "evaluation_split": manifest["evaluation_split"],
        "synthetic_demo": synthetic_demo,
        "formal_validity_decision": FORMAL_VALIDITY_DECISION,
        "manifest_sha256": _manifest_sha256(manifest),
        "provenance": dict(manifest["provenance"]),
        "arm_metrics": arm_metrics,
        "paired_comparisons": calculate_paired_comparisons(checked_attempts),
        "release_gate": {
            "release_ready": not blockers,
            "blocking_conditions": blockers,
            "human_review_approved": human_approved,
        },
        "evidence_boundary": (
            "Only independent Lean compilation establishes formal proof validity. "
            "SymPy evidence supports diagnostics and refutation search, not formal proof. "
            "Synthetic fixtures are workflow checks, never model-performance results."
        ),
        "attempts": [attempt.to_record() for attempt in checked_attempts],
    }


def load_attempts_jsonl(path: Path) -> list[FinalAttempt]:
    """Load normalized candidate-evaluation attempts from a JSONL file."""

    attempts: list[FinalAttempt] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"attempt record on line {line_number} must be an object")
        attempts.append(FinalAttempt.from_record(record))
    return attempts


def write_attempts_jsonl(path: Path, attempts: Sequence[FinalAttempt]) -> None:
    """Persist normalized attempts in deterministic JSONL form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(item.to_record(), sort_keys=True) for item in attempts)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_release_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    """Persist an auditable release bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of an immutable evidence file."""

    return sha256(path.read_bytes()).hexdigest()


def render_release_report(bundle: Mapping[str, Any]) -> str:
    """Render a compact report that keeps outcome and validity boundaries separate."""

    title = "Synthetic smoke report" if bundle["synthetic_demo"] else "Locked test evaluation report"
    lines = [
        f"# Phase 08 - {title}",
        "",
        f"- Run ID: {bundle['run_id']}",
        f"- Evaluation split: {bundle['evaluation_split']}",
        f"- Formal validity decision: {bundle['formal_validity_decision']}",
        f"- Release ready: {bundle['release_gate']['release_ready']}",
        "",
        "## Arm metrics",
        "",
        "| Arm | Compile@1 | Pass@k | Compile after allowed workflow | Refutation success |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in EXPECTED_ARMS:
        metric = bundle["arm_metrics"][arm]
        lines.append(
            "| {arm} | {compile_at_1:.3f} | {pass_at_k:.3f} | {after:.3f} | {refute:.3f} |".format(
                arm=arm,
                compile_at_1=metric["compile_at_1"],
                pass_at_k=metric["pass_at_k"],
                after=metric["compile_after_allowed_workflow"],
                refute=metric["refutation_success_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Release gate",
            "",
            *(
                [f"- Blocking condition: {item}" for item in bundle["release_gate"]["blocking_conditions"]]
                or ["- No release-gate blockers recorded."]
            ),
            "",
            "## Evidence boundary",
            "",
            bundle["evidence_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def write_release_report(path: Path, bundle: Mapping[str, Any]) -> None:
    """Write the human-readable companion to the JSON evidence bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_release_report(bundle), encoding="utf-8")
