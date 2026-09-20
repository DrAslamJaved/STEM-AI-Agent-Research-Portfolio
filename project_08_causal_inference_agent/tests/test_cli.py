import json
import sys

from causal_audit_agent.cli import main


def test_cli_emits_valid_causal_contract(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "causal-audit",
            "--question",
            "What is the effect of T on Y?",
            "--treatment",
            "T",
            "--outcome",
            "Y",
            "--estimand",
            "ATE",
            "--population",
            "eligible study population",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["treatment"] == "T"
    assert payload["outcome"] == "Y"
    assert payload["question_type"] == "causal"
