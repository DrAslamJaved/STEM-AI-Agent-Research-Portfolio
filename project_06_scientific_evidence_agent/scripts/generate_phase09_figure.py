"""Deterministic, dependency-free SVG generator for the Phase 9 trade-off figure.

Reads the frozen, committed Phase 8 result
(``results/controlled_experiments_dev.json``) and renders
``reports/figures/phase_09_evaluation_tradeoff.svg`` -- a plain hand-built SVG
with no external plotting library and no randomness, so re-running this
script against the same committed result byte-for-byte reproduces the same
figure. It never re-runs the experiment, downloads data, or trains a model;
it only reads one already-frozen, hash-verified JSON file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = PROJECT_ROOT / "results" / "controlled_experiments_dev.json"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "phase_09_evaluation_tradeoff.svg"

DISCLAIMER = "Held-out development evaluation — not an independent test."
DIRECT_COLOR = "#2f6f9f"
AUDITED_COLOR = "#c9722c"

_WIDTH = 820
_ROW_HEIGHT = 96
_CHART_LEFT = 260
_CHART_WIDTH = 380
_BAR_HEIGHT = 22
_BAR_GAP = 6


def extract_figure_data(report: Mapping[str, object]) -> dict[str, object]:
    """Pull only the fields this figure needs from the Phase 8 result report."""
    direct_metrics = report["direct_rag"]["audit_metrics"]
    audited_metrics = report["audited_agent"]["audit_metrics"]
    direct_official = report["direct_rag"]["official_scifact"]
    audited_official = report["audited_agent"]["official_scifact"]
    deltas = report["comparison_to_direct_rag"]["official_scifact"]
    bootstrap = report["official_bootstrap_confidence_intervals"]["metrics"]
    return {
        "coverage": {
            "direct_rag": float(direct_metrics["coverage"]),
            "audited_agent": float(audited_metrics["coverage"]),
        },
        "abstract_level_f1": {
            "direct_rag": float(direct_official["abstract_level"]["f1"]),
            "audited_agent": float(audited_official["abstract_level"]["f1"]),
            "delta": float(deltas["abstract_level_f1"]),
            "ci_lower": float(bootstrap["abstract_level_f1"]["lower"]),
            "ci_upper": float(bootstrap["abstract_level_f1"]["upper"]),
        },
        "sentence_level_f1": {
            "direct_rag": float(direct_official["sentence_level"]["f1"]),
            "audited_agent": float(audited_official["sentence_level"]["f1"]),
            "delta": float(deltas["sentence_level_f1"]),
            "ci_lower": float(bootstrap["sentence_level_f1"]["lower"]),
            "ci_upper": float(bootstrap["sentence_level_f1"]["upper"]),
        },
        "claim_count": int(direct_metrics["claim_count"]),
    }


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _bar_group(
    *,
    y_top: float,
    label: str,
    direct_value: float,
    audited_value: float,
    row_max: float,
    value_format: str,
    annotation: str | None,
) -> str:
    def bar(y: float, value: float, color: str, series_label: str) -> str:
        width = 0.0 if row_max <= 0 else max(0.0, value / row_max) * _CHART_WIDTH
        value_text = value_format.format(value)
        return (
            f'<rect x="{_CHART_LEFT}" y="{y:.1f}" width="{width:.1f}" height="{_BAR_HEIGHT}" '
            f'fill="{color}" />'
            f'<text x="{_CHART_LEFT + width + 8:.1f}" y="{y + _BAR_HEIGHT * 0.7:.1f}" '
            f'font-size="13" fill="#1a1a1a">{_escape(series_label)}: {_escape(value_text)}</text>'
        )

    direct_y = y_top
    audited_y = y_top + _BAR_HEIGHT + _BAR_GAP
    parts = [
        f'<text x="16" y="{y_top + _BAR_HEIGHT:.1f}" font-size="14" font-weight="600" '
        f'fill="#1a1a1a">{_escape(label)}</text>',
        bar(direct_y, direct_value, DIRECT_COLOR, "Direct RAG"),
        bar(audited_y, audited_value, AUDITED_COLOR, "Audited agent"),
    ]
    if annotation is not None:
        annotation_y = audited_y + _BAR_HEIGHT + 16
        parts.append(
            f'<text x="{_CHART_LEFT}" y="{annotation_y:.1f}" font-size="12" '
            f'fill="#444444">{_escape(annotation)}</text>'
        )
    return "\n    ".join(parts)


def build_svg(data: Mapping[str, object]) -> str:
    """Render the trade-off figure as a deterministic SVG string."""
    coverage = data["coverage"]
    abstract = data["abstract_level_f1"]
    sentence = data["sentence_level_f1"]
    claim_count = data["claim_count"]

    coverage_max = max(coverage["direct_rag"], coverage["audited_agent"], 1e-9) * 1.15
    f1_max = max(
        abstract["direct_rag"], abstract["audited_agent"],
        sentence["direct_rag"], sentence["audited_agent"],
        1e-9,
    ) * 1.2

    row_1_top = 96.0
    row_2_top = row_1_top + _ROW_HEIGHT
    row_3_top = row_2_top + _ROW_HEIGHT
    height = row_3_top + _ROW_HEIGHT + 40

    groups = [
        _bar_group(
            y_top=row_1_top,
            label="Coverage (not abstained)",
            direct_value=coverage["direct_rag"],
            audited_value=coverage["audited_agent"],
            row_max=coverage_max,
            value_format="{:.3f}",
            annotation=(
                "The audited agent trades coverage for citation discipline: "
                "it abstains far more often than direct RAG."
            ),
        ),
        _bar_group(
            y_top=row_2_top,
            label="Abstract-level F1 (official SciFact)",
            direct_value=abstract["direct_rag"],
            audited_value=abstract["audited_agent"],
            row_max=f1_max,
            value_format="{:.4f}",
            annotation=(
                f"Δ {abstract['delta']:+.4f}  "
                f"95% CI [{abstract['ci_lower']:+.4f}, {abstract['ci_upper']:+.4f}]"
            ),
        ),
        _bar_group(
            y_top=row_3_top,
            label="Sentence-level F1 (official SciFact)",
            direct_value=sentence["direct_rag"],
            audited_value=sentence["audited_agent"],
            row_max=f1_max,
            value_format="{:.4f}",
            annotation=(
                f"Δ {sentence['delta']:+.4f}  "
                f"95% CI [{sentence['ci_lower']:+.4f}, {sentence['ci_upper']:+.4f}]"
            ),
        ),
    ]

    svg_lines = "\n    ".join(groups)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {height:.0f}" '
        f'width="{_WIDTH}" height="{height:.0f}" role="img" '
        f'aria-label="Phase 8 audited-agent versus direct-RAG trade-off">\n'
        f'  <title>Phase 8 audited-agent vs direct-RAG trade-off</title>\n'
        f'  <rect x="0" y="0" width="{_WIDTH}" height="{height:.0f}" fill="#ffffff" />\n'
        f'  <text x="16" y="28" font-size="18" font-weight="700" fill="#111111">'
        f'Phase 8 evaluation trade-off: audited agent vs. direct RAG</text>\n'
        f'  <text x="16" y="50" font-size="14" font-weight="700" fill="#8a1f1f">'
        f'{_escape(DISCLAIMER)}</text>\n'
        f'  <text x="16" y="70" font-size="12" fill="#444444">'
        f'{claim_count} held-out development claims. Both arms consumed the same frozen runtime trace.</text>\n'
        f'  <rect x="{_CHART_LEFT}" y="80" width="14" height="14" fill="{DIRECT_COLOR}" />\n'
        f'  <text x="{_CHART_LEFT + 20}" y="91" font-size="12" fill="#1a1a1a">Direct RAG</text>\n'
        f'  <rect x="{_CHART_LEFT + 110}" y="80" width="14" height="14" fill="{AUDITED_COLOR}" />\n'
        f'  <text x="{_CHART_LEFT + 130}" y="91" font-size="12" fill="#1a1a1a">Audited agent</text>\n'
        f'    {svg_lines}\n'
        f'  <text x="16" y="{height - 14:.1f}" font-size="11" fill="#666666">'
        f'A positive delta is not a blanket superiority claim; interpret alongside coverage and faithfulness.</text>\n'
        f'</svg>\n'
    )


def main() -> int:
    report = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    data = extract_figure_data(report)
    svg_text = build_svg(data)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.write_text(svg_text, encoding="utf-8")
    print(f"Wrote {FIGURE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
