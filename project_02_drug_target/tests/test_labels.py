import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.labels import (
    LabelingError,
    add_binary_interaction_labels,
    summarize_binary_labels,
    write_label_summary,
    write_labeled_table,
)


def synthetic_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observed_pair_index": [0, 1, 2, 3, 4],
            "affinity_kd_nM": [10.0, 100.0, 100.1, 1000.0, 1000.1],
        }
    )


def test_add_binary_labels_preserves_source_and_threshold_boundaries() -> None:
    source = synthetic_interactions()

    labeled = add_binary_interaction_labels(source, [1000.0, 100.0])

    assert source.columns.tolist() == [
        "observed_pair_index",
        "affinity_kd_nM",
    ]
    assert labeled["interaction_kd_le_1000_nM"].tolist() == [1, 1, 1, 1, 0]
    assert labeled["interaction_kd_le_100_nM"].tolist() == [1, 1, 0, 0, 0]


def test_label_builder_rejects_invalid_affinities_and_thresholds() -> None:
    invalid_affinities = pd.DataFrame({"affinity_kd_nM": [10.0, 0.0]})

    with pytest.raises(LabelingError, match="greater than zero"):
        add_binary_interaction_labels(invalid_affinities)

    with pytest.raises(LabelingError, match="duplicates"):
        add_binary_interaction_labels(synthetic_interactions(), [100.0, 100.0])


def test_binary_label_summary_reports_expected_prevalence() -> None:
    labeled = add_binary_interaction_labels(synthetic_interactions(), [1000.0, 100.0])

    summaries = summarize_binary_labels(labeled, [1000.0, 100.0])

    assert summaries[0].label_column == "interaction_kd_le_1000_nM"
    assert summaries[0].positive_count == 4
    assert summaries[0].negative_count == 1
    assert summaries[0].positive_rate == pytest.approx(0.8)
    assert summaries[0].pKd_threshold == pytest.approx(6.0)

    assert summaries[1].label_column == "interaction_kd_le_100_nM"
    assert summaries[1].positive_count == 2
    assert summaries[1].negative_count == 3
    assert summaries[1].positive_rate == pytest.approx(0.4)
    assert summaries[1].pKd_threshold == pytest.approx(7.0)


def test_label_writers_create_reproducible_outputs(tmp_path: Path) -> None:
    labeled = add_binary_interaction_labels(synthetic_interactions())
    summaries = summarize_binary_labels(labeled)

    table_path = write_labeled_table(labeled, tmp_path / "labeled.csv")
    summary_path = write_label_summary(summaries, tmp_path / "summary.json")

    reloaded_table = pd.read_csv(table_path)
    report = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "interaction_kd_le_1000_nM" in reloaded_table.columns
    assert report["continuous_outcome"] == "pKd = -log10(Kd_nM / 1e9)"
    assert report["binary_label_variants"][0]["positive_count"] == 4