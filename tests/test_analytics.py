"""Tests for the analytics demo: CSV import, profiling, semantic model, query planning, tools."""

from __future__ import annotations

import anyio
import pytest

# Skip if duckdb not installed
duckdb = pytest.importorskip("duckdb")

from demos.analytics.src.analytics.agent import create_agent
from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.data_source import ColumnRole, Relationship
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.query_planner import Filter, QuerySpec, plan
from demos.analytics.src.analytics.semantic_model import Dimension, Metric, SemanticModel
from demos.analytics.src.analytics.semantic_roles import classify_role
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents import RequestContext, ToolEffect


@pytest.fixture
def sales_csv(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text(
        "date,region,product,amount,quantity\n"
        "2024-01-01,North,Widget,100,5\n"
        "2024-01-02,South,Gadget,200,10\n"
        "2024-01-03,North,Widget,150,7\n"
        "2024-01-04,East,Gadget,300,15\n"
        "2024-01-05,West,Widget,120,6\n"
        "2024-01-06,North,Gadget,250,12\n"
        "2024-01-07,South,Widget,180,9\n"
        "2024-01-08,East,Widget,90,4\n"
        "2024-01-09,West,Gadget,320,16\n"
        "2024-01-10,North,Widget,110,5\n"
    )
    return csv


@pytest.fixture
def source(sales_csv):
    s = CsvSource(named_csvs={"sales": sales_csv})
    yield s
    s.close()


def test_csv_source_imports_tables(source):
    tables = source.tables()
    assert len(tables) == 1
    assert tables[0].name == "sales"
    assert tables[0].rows == 10
    assert len(tables[0].columns) == 5


def test_csv_source_sample(source):
    rows = source.sample("sales", 3)
    assert len(rows) == 3
    assert "region" in rows[0]


def test_csv_source_native_query(source):
    rows = source.native_query("SELECT COUNT(*) AS n FROM sales")
    assert rows[0]["n"] == 10


def test_profile_dataset(source):
    profile = profile_dataset(source)
    assert len(profile.tables) == 1
    assert len(profile.columns) == 5
    # amount should be MEASURE_ADDITIVE
    amount_col = next(c for c in profile.columns if c.name == "amount")
    assert amount_col.rows == 10
    assert amount_col.distinct == 10
    assert amount_col.min == 90
    assert amount_col.max == 320


def test_classify_roles(source):
    profile = profile_dataset(source)
    roles = {c.name: classify_role(c) for c in profile.columns if c.table == "sales"}
    assert roles["amount"] == ColumnRole.MEASURE_ADDITIVE
    assert roles["quantity"] == ColumnRole.MEASURE_ADDITIVE
    assert roles["region"] == ColumnRole.DIMENSION
    assert roles["date"] == ColumnRole.DATE


def test_semantic_model(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    assert len(semantic.metrics) >= 2  # amount, quantity
    assert any(m.column == "amount" for m in semantic.metrics)
    assert any(d.column == "region" for d in semantic.dimensions)
    assert len(semantic.time_columns) >= 1


def test_query_planner_single_table(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    sql = plan(
        semantic,
        QuerySpec(
            metrics=("sales.amount",),
            dimensions=("sales.region",),
            limit=5,
        ),
    )
    rows = source.native_query_with_limit(sql, 10)
    assert len(rows) == 4  # 4 distinct regions
    assert "amount" in rows[0]


def test_query_planner_aggregation(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    sql = plan(semantic, QuerySpec(metrics=("sales.amount",)))
    rows = source.native_query_with_limit(sql, 10)
    assert rows[0]["amount"] == 1820  # total sum


def test_query_planner_with_filter(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    sql = plan(
        semantic,
        QuerySpec(
            metrics=("sales.amount",),
            dimensions=("sales.region",),
            filters=(Filter(column="sales.region", op="=", value="North"),),
        ),
    )
    rows = source.native_query_with_limit(sql, 10)
    assert len(rows) == 1
    assert rows[0]["region"] == "North"


def test_toolset_describe_dataset(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.describe_dataset()
        assert tool.spec.effect == ToolEffect.READ_ONLY
        result = await tool.invoke({}, RequestContext.ephemeral())
        assert not result.error
        assert "sales" in result.content

    anyio.run(run)


def test_toolset_run_query(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.run_query()
        result = await tool.invoke(
            {"metrics": ["sales.amount"], "dimensions": ["sales.region"], "limit": 5},
            RequestContext.ephemeral(),
        )
        assert not result.error
        assert "North" in result.content or "South" in result.content

    anyio.run(run)


def test_toolset_summarize(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.summarize()
        result = await tool.invoke({"metric": "sales.amount"}, RequestContext.ephemeral())
        assert not result.error

    anyio.run(run)


def test_toolset_run_sql_blocks_writes(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.run_sql()
        result = await tool.invoke({"sql": "DELETE FROM sales"}, RequestContext.ephemeral())
        assert result.error
        assert "forbidden" in result.content.lower()

    anyio.run(run)


def test_toolset_run_sql_does_not_block_keyword_substrings(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.run_sql()
        result = await tool.invoke(
            {"sql": "SELECT date AS created_at FROM sales LIMIT 1"},
            RequestContext.ephemeral(),
        )
        assert not result.error

    anyio.run(run)


def test_toolset_run_sql_blocks_multiple_statements(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.run_sql()
        result = await tool.invoke(
            {"sql": "SELECT * FROM sales LIMIT 1; DROP TABLE sales"},
            RequestContext.ephemeral(),
        )
        assert result.error
        assert "multiple" in result.content.lower()

    anyio.run(run)


def test_toolset_run_sql_allows_reads(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        tool = tools.run_sql()
        result = await tool.invoke(
            {"sql": "SELECT region, COUNT(*) AS n FROM sales GROUP BY region"},
            RequestContext.ephemeral(),
        )
        assert not result.error

    anyio.run(run)


def test_toolset_all_tools_read_only(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)
    for tool in tools.all_tools():
        assert tool.spec.effect == ToolEffect.READ_ONLY
        assert tool.spec.input_schema["type"] == "object"
        assert "properties" in tool.spec.input_schema


def test_create_agent_wires_unique_tool_names(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    agent = create_agent(source, object(), semantic)
    names = [tool.spec.name for tool in agent.tools]
    assert len(names) == len(set(names))
    assert "run_query" in names
    assert "run_sql" in names
    # sprawl removed: no ML/advanced tools are wired into the agent
    assert "cohort_analysis" not in names
    assert "ml_regression" not in names


def test_multi_fact_planning_fails_instead_of_cross_joining():
    semantic = SemanticModel(
        metrics=(
            Metric(table="fact_a", column="amount", aggregation="sum"),
            Metric(table="fact_b", column="cost", aggregation="sum"),
        ),
        dimensions=(
            Dimension(table="fact_a", column="region"),
            Dimension(table="fact_b", column="region"),
        ),
        entity_keys=(),
        time_columns=(),
        relationships=(
            Relationship(
                from_table="fact_a",
                from_columns=("region",),
                to_table="fact_b",
                to_columns=("region",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="multi-fact metric queries"):
        plan(semantic, QuerySpec(metrics=("fact_a.amount", "fact_b.cost")))
