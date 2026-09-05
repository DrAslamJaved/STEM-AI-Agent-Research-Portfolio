"""Tests for reproducible train-only lexical verification models."""

from __future__ import annotations

from evidence_agent.data.schemas import Verdict
from evidence_agent.data.acquisition import sha256_file
from evidence_agent.verification.models import (
    SentenceInput,
    StanceInput,
    fit_verifier_bundle,
    load_verifier_bundle,
    write_verifier_bundle,
)
from evidence_agent.verification.scifact import load_verification_training_data
from tests.helpers import write_verification_scifact_dataset


def _bundle(dataset):
    training = load_verification_training_data(
        dataset / "claims_train.jsonl", dataset / "corpus.jsonl"
    )
    return fit_verifier_bundle(
        training.stance_examples,
        training.sentence_examples,
        training_claims_sha256=sha256_file(dataset / "claims_train.jsonl"),
        corpus_sha256=sha256_file(dataset / "corpus.jsonl"),
        max_features=100,
    )


def test_verifier_bundle_scores_all_stance_classes_and_sentence_probability(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")
    bundle = _bundle(dataset)

    prediction = bundle.predict_stances(
        (
            StanceInput(
                claim_id=4,
                doc_id=10,
                claim_text="Aspirin reduces inflammation.",
                document_text="Inflammation study Aspirin reduces inflammation in the study population.",
            ),
        )
    )[0]
    sentence_score = bundle.score_sentences(
        (
            SentenceInput(
                claim_id=4,
                doc_id=10,
                sentence_id=0,
                claim_text="Aspirin reduces inflammation.",
                sentence_text="Aspirin reduces inflammation in the study population.",
            ),
        )
    )[0]

    assert {label for label, _ in prediction.probabilities} == {
        Verdict.SUPPORT,
        Verdict.CONTRADICT,
        Verdict.NO_EVIDENCE,
    }
    assert 0.0 <= prediction.confidence <= 1.0
    assert 0.0 <= sentence_score.probability <= 1.0


def test_verifier_bundle_round_trip_preserves_predictions(tmp_path) -> None:
    dataset = write_verification_scifact_dataset(tmp_path / "scifact")
    bundle = _bundle(dataset)
    model_path = tmp_path / "artifacts" / "verifier.joblib"
    input_pair = StanceInput(
        claim_id=4,
        doc_id=10,
        claim_text="Aspirin reduces inflammation.",
        document_text="Inflammation study Aspirin reduces inflammation in the study population.",
    )

    write_verifier_bundle(bundle, model_path)
    restored = load_verifier_bundle(model_path)

    assert restored.summary_dict()["format"] == "evidence_agent_lexical_relation_verifier_v2"
    assert restored.predict_stances((input_pair,)) == bundle.predict_stances((input_pair,))
