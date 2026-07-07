"""Tests for advanced analytics tools: cohort, funnel, RFM, decomposition, etc."""

from __future__ import annotations

import csv
import random
import tempfile
from pathlib import Path

import anyio
import pytest

duckdb = pytest.importorskip("duckdb")
sklearn = pytest.importorskip("sklearn")
statsmodels = pytest.importorskip("statsmodels")
scipy = pytest.importorskip("scipy")

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.advanced_tools import AdvancedToolset
from python_ai_agents import RequestContext, ToolEffect


@pytest.fixture
def multi_date_data(tmp_path):
    """Create player visit data spanning multiple days for cohort/RFM/time-series tests."""
    random.seed(42)
    csv_path = tmp_path / "visits.csv"
    rows = []
    players = [f"player_{i}" for i in range(100)]
    from datetime import date, timedelta
    start = date(2024, 1, 1)
    for p in players:
        reg_day = random.randint(0, 60)
        n_visits = random.randint(1, 20)
        for v in range(n_visits):
            day = start + timedelta(days=reg_day + v * random.randint(1, 30))
            rows.append({
                "playerId": p,
                "day": day.isoformat(),
                "coinIn": round(random.gauss(500, 200), 2),
                "netWin": round(random.gauss(-50, 30), 2),
                "group": random.choice(["A", "B"]),
                "treatment": random.choice([0, 1]),
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["playerId", "day", "coinIn", "netWin", "group", "treatment"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@pytest.fixture
def adv_setup(multi_date_data):
    source = CsvSource(named_csvs={"visits": multi_date_data})
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    yield source, semantic
    source.close()


def test_cohort_analysis(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.cohort_analysis().invoke(
            {"entityColumn": "playerId", "timeColumn": "visits.day", "cohortGrain": "month"},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "cohort" in result.content.lower()

    anyio.run(run)


def test_funnel_analysis(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.funnel_analysis().invoke(
            {"entityColumn": "playerId",
             "steps": [
                 {"name": "All visitors", "filters": []},
                 {"name": "High spenders", "filters": [{"column": "visits.coinIn", "op": ">", "value": "400"}]},
             ]},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "funnel" in result.content.lower()

    anyio.run(run)


def test_rfm_segmentation(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.rfm_segmentation().invoke(
            {"table": "visits", "entityColumn": "playerId",
             "timeColumn": "visits.day", "monetaryColumn": "visits.coinIn"},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "rfm" in result.content.lower() or "segment" in result.content.lower()

    anyio.run(run)


def test_time_series_decomposition(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.time_series_decomposition().invoke(
            {"metric": "visits.coinIn", "timeColumn": "visits.day", "period": 7},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "trend" in result.content.lower()

    anyio.run(run)


def test_correlation_matrix(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.correlation_matrix().invoke(
            {"table": "visits", "columns": ["coinIn", "netWin"]},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "matrix" in result.content.lower() or "correlation" in result.content.lower()

    anyio.run(run)


def test_data_quality(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.data_quality().invoke(
            {"table": "visits"}, RequestContext.ephemeral(),
        )
        assert not result.error
        assert "quality" in result.content.lower()

    anyio.run(run)


def test_percentile_ranking(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.percentile_ranking().invoke(
            {"table": "visits", "entityColumn": "playerId", "metric": "visits.coinIn"},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "percentile" in result.content.lower()

    anyio.run(run)


def test_benchmark_comparison(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        # Get a player ID first
        rows = source.native_query("SELECT DISTINCT playerId FROM visits LIMIT 1")
        target = rows[0]["playerId"]
        result = await adv.benchmark_comparison().invoke(
            {"table": "visits", "entityColumn": "playerId",
             "metric": "visits.coinIn", "targetEntity": target},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "benchmark" in result.content.lower() or "z_score" in result.content.lower()

    anyio.run(run)


def test_pca_analysis(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.pca_analysis().invoke(
            {"table": "visits", "columns": ["coinIn", "netWin"], "nComponents": 2},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "explained_variance" in result.content.lower() or "pca" in result.content.lower()

    anyio.run(run)


def test_survival_analysis(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.survival_analysis().invoke(
            {"entityColumn": "playerId", "timeColumn": "visits.day"},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "survival" in result.content.lower() or "duration" in result.content.lower()

    anyio.run(run)


def test_granger_causality(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)

    async def run():
        result = await adv.granger_causality().invoke(
            {"cause": "visits.coinIn", "effect": "visits.netWin",
             "timeColumn": "visits.day", "maxLag": 3},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "granger" in result.content.lower() or "causality" in result.content.lower()

    anyio.run(run)


def test_all_advanced_tools_are_read_only(adv_setup):
    source, semantic = adv_setup
    adv = AdvancedToolset(source, semantic)
    for tool in adv.all_tools():
        assert tool.spec.effect == ToolEffect.READ_ONLY
        assert tool.spec.input_schema["type"] == "object"
        assert "properties" in tool.spec.input_schema


def test_advanced_tool_count():
    """Verify we have 11 advanced tools."""
    from demos.analytics.src.analytics.csv_source import CsvSource
    tmp = Path(tempfile.mkdtemp())
    csv = tmp / "t.csv"
    csv.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14\n15,16\n17,18\n19,20\n")
    source = CsvSource(named_csvs={"t": csv})
    semantic = SemanticModel.from_profile(profile_dataset(source))
    adv = AdvancedToolset(source, semantic)
    tools = adv.all_tools()
    assert len(tools) == 11
    names = {t.spec.name for t in tools}
    assert names == {
        "cohort_analysis", "funnel_analysis", "rfm_segmentation",
        "time_series_decomposition", "correlation_matrix", "data_quality",
        "percentile_ranking", "benchmark_comparison", "pca_analysis",
        "survival_analysis", "granger_causality",
    }
    source.close()
