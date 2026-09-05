"""Tests for the evaluator/runtime leakage boundary."""

from __future__ import annotations

import pytest

from evidence_agent.contracts import (
    LeakageContractError,
    assert_runtime_payload_is_safe,
    runtime_claim_from_scifact,
)


def test_runtime_claim_discards_gold_scifact_annotations() -> None:
    raw_claim = {
        "id": 7,
        "claim": "A controlled scientific claim.",
        "evidence": {"123": [{"label": "SUPPORT", "sentences": [0]}]},
        "cited_doc_ids": [123],
    }

    runtime_claim = runtime_claim_from_scifact(raw_claim)

    assert runtime_claim.claim_id == 7
    assert runtime_claim.text == "A controlled scientific claim."
    assert not hasattr(runtime_claim, "evidence")
    assert not hasattr(runtime_claim, "cited_doc_ids")


@pytest.mark.parametrize("field", ["evidence", "cited_doc_ids"])
def test_runtime_payload_rejects_gold_annotations(field: str) -> None:
    with pytest.raises(LeakageContractError, match=field):
        assert_runtime_payload_is_safe({"id": 7, "claim": "Safe text", field: []})
