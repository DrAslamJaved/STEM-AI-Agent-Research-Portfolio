"""Offline evaluators that may access frozen SciFact gold annotations."""

from evidence_agent.evaluation.retrieval import RetrievalEvaluationResult
from evidence_agent.evaluation.verification import EvidenceVerificationResult, StanceBenchmarkResult

__all__ = ["EvidenceVerificationResult", "RetrievalEvaluationResult", "StanceBenchmarkResult"]
