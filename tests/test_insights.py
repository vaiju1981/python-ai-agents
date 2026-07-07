"""Tests for deterministic dataset insights (DuckDB-backed)."""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.insights import generate_insights
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel


@pytest.fixture
def source(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "date,region,product,amount\n"
        "2024-01-01,North,Widget,100\n"
        "2024-01-02,South,Gadget,200\n"
        "2024-02-01,North,Widget,150\n"
        "2024-02-02,East,Gadget,300\n"
        "2024-03-01,West,Widget,120\n"
        "2024-03-02,North,Gadget,250\n"
    )
    s = CsvSource(named_csvs={"sales": csv})
    yield s
    s.close()


def test_overview_and_metric_insight(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    assert semantic.metrics, "amount should be detected as a measure"

    insights = generate_insights(source, semantic, profile)
    assert insights[0].title == "Dataset overview"
    metric_col = semantic.metrics[0].column
    assert any(metric_col in i.title for i in insights)


def test_charted_insights_carry_their_rows(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    insights = generate_insights(source, semantic, profile)

    charted = [i for i in insights if i.chart is not None]
    assert charted, "expected at least one insight with a chart"
    for ins in charted:
        assert ins.rows, "a charted insight must carry the rows behind its chart"
        assert ins.chart.x  # a real axis column
