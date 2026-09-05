"""Train-only lexical NLI and sentence-evidence models for SciFact.

The persisted bundle is a trusted, local experiment artifact.  It must be
created by :func:`fit_verifier_bundle`; loading an artifact from an untrusted
source is unsafe because ``joblib`` uses Python object deserialization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evidence_agent.data.schemas import Verdict


VERIFIER_BUNDLE_FORMAT = "evidence_agent_lexical_relation_verifier_v2"
DEFAULT_RANDOM_SEED = 20260904
DEFAULT_MAX_FEATURES = 40_000


class VerificationModelError(ValueError):
    """Raised when training, loading, or scoring a verifier is invalid."""


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VerificationModelError(f"{name} must be a positive integer.")


def _validate_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VerificationModelError(f"{name} must be a non-negative integer.")


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise VerificationModelError(f"{name} must be non-empty text.")


def _pair_text(claim_text: str, evidence_text: str) -> str:
    """Format a deterministic text pair for the lexical models."""
    _validate_text(claim_text, "claim_text")
    _validate_text(evidence_text, "evidence_text")
    return f"claim: {claim_text}\n[SEP]\nevidence: {evidence_text}"


def _relation_matrix(
    vectorizer: TfidfVectorizer, claim_texts: Sequence[str], evidence_texts: Sequence[str]
):
    """Encode pair direction and lexical interaction, not just text concatenation.

    Features are claim TF-IDF, evidence TF-IDF, element-wise overlap, and the
    absolute TF-IDF difference.  This gives the linear model explicit access to
    claim/evidence relation signals while remaining deterministic and local.
    """
    if len(claim_texts) != len(evidence_texts):
        raise VerificationModelError("Claim and evidence text counts must match.")
    claim_matrix = vectorizer.transform(claim_texts)
    evidence_matrix = vectorizer.transform(evidence_texts)
    return hstack(
        (
            claim_matrix,
            evidence_matrix,
            claim_matrix.multiply(evidence_matrix),
            abs(claim_matrix - evidence_matrix),
        ),
        format="csr",
    )


@dataclass(frozen=True, slots=True)
class StanceInput:
    """Runtime-safe claim/document input to the three-way stance classifier."""

    claim_id: int
    doc_id: int
    claim_text: str
    document_text: str

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.claim_id, "claim_id")
        _validate_non_negative_int(self.doc_id, "doc_id")
        _validate_text(self.claim_text, "claim_text")
        _validate_text(self.document_text, "document_text")


@dataclass(frozen=True, slots=True)
class StanceTrainingExample:
    """One labelled train-split example for fitting the stance model."""

    input: StanceInput
    label: Verdict


@dataclass(frozen=True, slots=True)
class SentenceInput:
    """Runtime-safe claim/sentence input to the evidence selector."""

    claim_id: int
    doc_id: int
    sentence_id: int
    claim_text: str
    sentence_text: str

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.claim_id, "claim_id")
        _validate_non_negative_int(self.doc_id, "doc_id")
        _validate_non_negative_int(self.sentence_id, "sentence_id")
        _validate_text(self.claim_text, "claim_text")
        _validate_text(self.sentence_text, "sentence_text")


@dataclass(frozen=True, slots=True)
class SentenceTrainingExample:
    """One binary labelled train-split example for fitting the selector."""

    input: SentenceInput
    is_evidence: bool

    def __post_init__(self) -> None:
        if not isinstance(self.is_evidence, bool):
            raise VerificationModelError("is_evidence must be boolean.")


@dataclass(frozen=True, slots=True)
class StancePrediction:
    """A three-way NLI-style prediction with all class probabilities retained."""

    input: StanceInput
    verdict: Verdict
    probabilities: tuple[tuple[Verdict, float], ...]

    @property
    def confidence(self) -> float:
        """Return the probability assigned to the emitted class."""
        return self.probability(self.verdict)

    def probability(self, verdict: Verdict) -> float:
        """Return a named class probability, including ``NO_EVIDENCE``."""
        for candidate, value in self.probabilities:
            if candidate is verdict:
                return value
        return 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.input.claim_id,
            "confidence": self.confidence,
            "doc_id": self.input.doc_id,
            "probabilities": {str(label): value for label, value in self.probabilities},
            "verdict": str(self.verdict),
        }


@dataclass(frozen=True, slots=True)
class SentenceScore:
    """Probability that a public corpus sentence is claim evidence."""

    input: SentenceInput
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise VerificationModelError("Sentence evidence probability must lie in [0, 1].")

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.input.claim_id,
            "doc_id": self.input.doc_id,
            "evidence_probability": self.probability,
            "sentence_id": self.input.sentence_id,
        }


@dataclass(frozen=True, slots=True)
class VerifierBundle:
    """A locally persisted lexical stance model plus sentence selector."""

    stance_vectorizer: TfidfVectorizer
    stance_classifier: LogisticRegression
    sentence_vectorizer: TfidfVectorizer
    sentence_classifier: LogisticRegression
    random_seed: int
    max_features: int
    training_claims_sha256: str
    corpus_sha256: str
    stance_label_counts: tuple[tuple[Verdict, int], ...]
    sentence_example_count: int

    def __post_init__(self) -> None:
        _validate_positive_int(self.random_seed, "random_seed")
        _validate_positive_int(self.max_features, "max_features")
        _validate_positive_int(self.sentence_example_count, "sentence_example_count")
        for name, digest in (
            ("training_claims_sha256", self.training_claims_sha256),
            ("corpus_sha256", self.corpus_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise VerificationModelError(f"{name} must be a SHA-256 digest.")
        expected_labels = {Verdict.SUPPORT, Verdict.CONTRADICT, Verdict.NO_EVIDENCE}
        observed_labels = {label for label, _ in self.stance_label_counts}
        if observed_labels != expected_labels:
            raise VerificationModelError(
                "The stance model must retain SUPPORT, CONTRADICT, and NO_EVIDENCE training data."
            )

    def summary_dict(self) -> dict[str, object]:
        """Return enough metadata to reproduce a locally trained artifact."""
        return {
            "algorithm": "tfidf_relation_features_logistic_regression_stance_and_sentence_selector",
            "corpus_sha256": self.corpus_sha256,
            "format": VERIFIER_BUNDLE_FORMAT,
            "max_features": self.max_features,
            "random_seed": self.random_seed,
            "scikit_learn_version": sklearn.__version__,
            "sentence_example_count": self.sentence_example_count,
            "sentence_vocabulary_size": len(self.sentence_vectorizer.vocabulary_),
            "stance_label_counts": {
                str(label): count for label, count in self.stance_label_counts
            },
            "stance_vocabulary_size": len(self.stance_vectorizer.vocabulary_),
            "training_claims_sha256": self.training_claims_sha256,
        }

    def predict_stances(self, inputs: Sequence[StanceInput]) -> tuple[StancePrediction, ...]:
        """Classify safe claim/document inputs without access to gold annotations."""
        if not inputs:
            return ()
        probabilities = self.stance_classifier.predict_proba(
            _relation_matrix(
                self.stance_vectorizer,
                [item.claim_text for item in inputs],
                [item.document_text for item in inputs],
            )
        )
        raw_classes = tuple(str(label) for label in self.stance_classifier.classes_)
        class_labels = tuple(Verdict(label) for label in raw_classes)
        expected_labels = {Verdict.SUPPORT, Verdict.CONTRADICT, Verdict.NO_EVIDENCE}
        if set(class_labels) != expected_labels:
            raise VerificationModelError("Persisted stance classifier has unexpected classes.")

        predictions: list[StancePrediction] = []
        for item, row in zip(inputs, probabilities, strict=True):
            index = int(np.argmax(row))
            predictions.append(
                StancePrediction(
                    input=item,
                    verdict=class_labels[index],
                    probabilities=tuple(
                        (label, float(probability))
                        for label, probability in zip(class_labels, row, strict=True)
                    ),
                )
            )
        return tuple(predictions)

    def score_sentences(self, inputs: Sequence[SentenceInput]) -> tuple[SentenceScore, ...]:
        """Score safe claim/sentence inputs without consulting gold rationales."""
        if not inputs:
            return ()
        probabilities = self.sentence_classifier.predict_proba(
            _relation_matrix(
                self.sentence_vectorizer,
                [item.claim_text for item in inputs],
                [item.sentence_text for item in inputs],
            )
        )
        classes = tuple(int(label) for label in self.sentence_classifier.classes_)
        if set(classes) != {0, 1}:
            raise VerificationModelError("Persisted sentence classifier must have binary classes.")
        evidence_index = classes.index(1)
        return tuple(
            SentenceScore(input=item, probability=float(row[evidence_index]))
            for item, row in zip(inputs, probabilities, strict=True)
        )


def _build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        min_df=1,
        ngram_range=(1, 2),
        norm="l2",
        strip_accents="unicode",
        sublinear_tf=True,
    )


def _validate_examples(
    stance_examples: Sequence[StanceTrainingExample],
    sentence_examples: Sequence[SentenceTrainingExample],
) -> None:
    if not stance_examples:
        raise VerificationModelError("At least one stance training example is required.")
    if not sentence_examples:
        raise VerificationModelError("At least one sentence training example is required.")
    expected_labels = {Verdict.SUPPORT, Verdict.CONTRADICT, Verdict.NO_EVIDENCE}
    observed_labels = {example.label for example in stance_examples}
    if observed_labels != expected_labels:
        missing = ", ".join(str(label) for label in sorted(expected_labels - observed_labels))
        raise VerificationModelError(f"Stance training is missing class(es): {missing}.")
    if {example.is_evidence for example in sentence_examples} != {False, True}:
        raise VerificationModelError("Sentence training requires both evidence and non-evidence examples.")


def fit_verifier_bundle(
    stance_examples: Sequence[StanceTrainingExample],
    sentence_examples: Sequence[SentenceTrainingExample],
    *,
    training_claims_sha256: str,
    corpus_sha256: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_features: int = DEFAULT_MAX_FEATURES,
) -> VerifierBundle:
    """Fit deterministic lexical verification models using train-split labels only."""
    _validate_positive_int(random_seed, "random_seed")
    _validate_positive_int(max_features, "max_features")
    _validate_examples(stance_examples, sentence_examples)

    stance_vectorizer = _build_vectorizer(max_features)
    stance_claim_texts = [example.input.claim_text for example in stance_examples]
    stance_document_texts = [example.input.document_text for example in stance_examples]
    stance_vectorizer.fit([*stance_claim_texts, *stance_document_texts])
    stance_matrix = _relation_matrix(
        stance_vectorizer,
        stance_claim_texts,
        stance_document_texts,
    )
    stance_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_seed,
    )
    stance_classifier.fit(stance_matrix, [str(example.label) for example in stance_examples])

    sentence_vectorizer = _build_vectorizer(max_features)
    sentence_claim_texts = [example.input.claim_text for example in sentence_examples]
    sentence_texts = [example.input.sentence_text for example in sentence_examples]
    sentence_vectorizer.fit([*sentence_claim_texts, *sentence_texts])
    sentence_matrix = _relation_matrix(
        sentence_vectorizer,
        sentence_claim_texts,
        sentence_texts,
    )
    sentence_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_seed,
    )
    sentence_classifier.fit(sentence_matrix, [int(example.is_evidence) for example in sentence_examples])

    label_counts = Counter(example.label for example in stance_examples)
    return VerifierBundle(
        stance_vectorizer=stance_vectorizer,
        stance_classifier=stance_classifier,
        sentence_vectorizer=sentence_vectorizer,
        sentence_classifier=sentence_classifier,
        random_seed=random_seed,
        max_features=max_features,
        training_claims_sha256=training_claims_sha256,
        corpus_sha256=corpus_sha256,
        stance_label_counts=tuple((label, label_counts[label]) for label in Verdict),
        sentence_example_count=len(sentence_examples),
    )


def write_verifier_bundle(bundle: VerifierBundle, path: Path) -> None:
    """Persist a locally generated verifier bundle under an ignored artifact path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_verifier_bundle(path: Path) -> VerifierBundle:
    """Load a trusted local verifier artifact created by this package."""
    path = Path(path)
    try:
        bundle = joblib.load(path)
    except (EOFError, OSError, ValueError) as error:
        raise VerificationModelError(f"Unable to read verifier bundle {path}: {error}") from error
    if not isinstance(bundle, VerifierBundle):
        raise VerificationModelError("Verifier artifact has an unexpected type.")
    return bundle
