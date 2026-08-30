from pathlib import Path

import yaml


REQUIRED_HEADINGS = (
    "# Phase 9 Agent Reasoning Response",
    "## Decision",
    "## 1. Geometric subspace argument",
    "## 2. Anomaly evidence is not proof",
    "## 3. Normal-training assumption",
    "## 4. Leakage and contamination controls",
    "## 5. Scaling and threshold selection",
    "## 6. False-positive and false-negative consequences",
    "## 7. Empirical validation and next action",
    "## Evidence sources",
)

REQUIRED_FRAGMENTS = (
    "42,000 normal fitting observations",
    "14,000 normal calibration observations",
    "82,332 official test observations",
    "64 standardized model features",
    "34 principal components",
    "0.9521414327676875",
    "0.4923769885740442",
    "0.05195005735462808",
    "0.9480499426453719",
    "42,977 false negatives",
    "1,026 false positives",
    "not proof that the flow is malicious",
    "not recommended for operational deployment",
    "post-evaluation tuning performed: 0",
)


def test_phase_nine_reasoning_response_contract() -> None:
    configuration = yaml.safe_load(
        Path("configs/baseline.yaml").read_text(
            encoding="utf-8"
        )
    )

    response_path = Path(
        configuration[
            "agent_reasoning_evaluation"
        ]["response_path"]
    )

    assert response_path == Path(
        "results/"
        "phase_09_agent_reasoning_response.md"
    )
    assert response_path.is_file()
    assert response_path.stat().st_size > 0

    data = response_path.read_bytes()

    assert not data.startswith(b"\xef\xbb\xbf")
    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")

    response = data.decode("utf-8")

    for heading in REQUIRED_HEADINGS:
        assert heading in response

    for fragment in REQUIRED_FRAGMENTS:
        assert fragment in response

    assert (
        response.count(
            "not recommended for operational deployment"
        )
        == 1
    )
