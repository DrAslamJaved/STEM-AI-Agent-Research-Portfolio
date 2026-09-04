"""CLI behaviour tests for the foundation phase."""

from __future__ import annotations

import json

from evidence_agent.cli import main


def test_contract_command_prints_machine_readable_project_contract(capsys) -> None:
    assert main(["contract"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["current_phase"] == "foundation"
    assert payload["runtime_gold_fields_forbidden"] == ["evidence", "cited_doc_ids"]


def test_future_command_fails_clearly_until_its_phase_is_implemented(capsys) -> None:
    assert main(["validate-data"]) == 2
    assert "not available yet" in capsys.readouterr().err
