from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    PREDICTIVE = "predictive"
    ASSOCIATIONAL = "associational"
    CAUSAL = "causal"
    AMBIGUOUS = "ambiguous"


class AnalysisState(str, Enum):
    DRAFT = "DRAFT"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    IDENTIFIED_AND_ROBUST = "IDENTIFIED_AND_ROBUST"
    IDENTIFIED_BUT_FRAGILE = "IDENTIFIED_BUT_FRAGILE"
    IDENTIFIED_BUT_NOT_ESTIMABLE = "IDENTIFIED_BUT_NOT_ESTIMABLE"
    NON_IDENTIFIABLE = "NON_IDENTIFIABLE"
    INSUFFICIENT_ASSUMPTIONS = "INSUFFICIENT_ASSUMPTIONS"
    INVALID_CAUSAL_QUERY = "INVALID_CAUSAL_QUERY"


@dataclass(frozen=True)
class CausalQuestion:
    question: str
    treatment: str
    outcome: str
    estimand: str = "ATE"
    population: str = "study population"
    treatment_time: str | None = None
    outcome_time: str | None = None
    question_type: QuestionType = QuestionType.CAUSAL

    def validate(self) -> None:
        missing = [
            name
            for name in ("question", "treatment", "outcome", "estimand", "population")
            if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise ValueError(f"Missing required causal fields: {', '.join(missing)}")
        if self.treatment == self.outcome:
            raise ValueError("Treatment and outcome must be distinct variables")
        if self.estimand not in {"ATE", "ATT", "CATE"}:
            raise ValueError(f"Unsupported estimand: {self.estimand}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["question_type"] = self.question_type.value
        return data
