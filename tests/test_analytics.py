"""Tests for the analytics demo: CSV import, profiling, semantic model, query planning, tools."""

from __future__ import annotations

import anyio
import pytest

# Skip if duckdb not installed
duckdb = pytest.importorskip("duckdb")

from demos.analytics.src.analytics.agent import create_agent
from demos.analytics.src.analytics.charts import choose_chart
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


def test_multi_fact_planning_joins_facts():
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

    sql = plan(semantic, QuerySpec(metrics=("fact_a.amount", "fact_b.cost")))
    assert "JOIN" in sql
    assert "fact_a" in sql and "fact_b" in sql
    # Each fact is pre-aggregated in its own CTE.
    assert "WITH" in sql


@pytest.fixture
def two_fact_source(tmp_path):
    sales = tmp_path / "sales.csv"
    sales.write_text("region,amount\nNorth,100\nSouth,200\nNorth,50\n")
    returns = tmp_path / "returns.csv"
    returns.write_text("region,cost\nNorth,10\nSouth,25\n")
    s = CsvSource(named_csvs={"sales": sales, "returns": returns})
    yield s
    s.close()


def test_multi_fact_query_executes(two_fact_source):
    profile = profile_dataset(two_fact_source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(two_fact_source, semantic)

    async def run():
        result = await tools.run_query().invoke(
            {
                "metrics": ["sales.amount", "returns.cost"],
                "dimensions": ["sales.region"],
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        rows = result.data
        by_region = {r["region"]: r for r in rows}
        assert by_region["North"]["amount"] == 150
        assert by_region["North"]["cost"] == 10
        assert by_region["South"]["amount"] == 200
        assert by_region["South"]["cost"] == 25

    anyio.run(run)


def test_compare_returns_chartable_rows(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        result = await tools.compare().invoke(
            {
                "metrics": ["sales.amount"],
                "timeColumn": "sales.date",
                "lastDays": 10000,
                "dimensions": ["sales.region"],
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        assert isinstance(result.data, list) and result.data
        row = result.data[0]
        assert "region" in row
        assert "amount" in row
        assert "amount_prev" in row

    anyio.run(run)


def test_choose_chart_compare_rows():
    rows = [
        {"region": "North", "amount": 150, "amount_prev": 120},
        {"region": "South", "amount": 200, "amount_prev": 180},
    ]
    spec = choose_chart(rows)
    assert spec is not None
    assert spec.kind == "bar"
    assert spec.x == "region"


def test_choose_chart_forecast_rows():
    rows = [
        {"step": 1, "value": 10, "low": 8, "high": 12},
        {"step": 2, "value": 12, "low": 9, "high": 15},
    ]
    spec = choose_chart(rows)
    assert spec is not None
    assert spec.kind == "line"
    assert spec.x == "step"
    assert spec.y == "value"


def test_extract_rows_from_dict_payload():
    from demos.analytics.src.analytics.app import _extract_rows

    payload = {"current": [{"region": "North", "amount": 150}], "previous": []}
    rows = _extract_rows(payload)
    assert rows == [{"region": "North", "amount": 150}]


def test_query_planner_derived_metric_ratio(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    sql = plan(
        semantic,
        QuerySpec(
            metrics=("sales.amount",),
            dimensions=("sales.region",),
            derivedMetrics=({"name": "avg_price", "expression": "sales.amount/sales.quantity"},),
        ),
    )
    assert "SUM(" in sql
    assert '"sales"."amount"' in sql and '"sales"."quantity"' in sql
    rows = source.native_query_with_limit(sql, 10)
    # North: amount 100+150+250+110=610, quantity 5+7+12+5=29 -> ~21.034
    north = next(r for r in rows if r["region"] == "North")
    assert float(north["avg_price"]) == pytest.approx(610 / 29)


def test_run_query_derived_metric(source):
    profile = profile_dataset(source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(source, semantic)

    async def run():
        result = await tools.run_query().invoke(
            {
                "metrics": ["sales.amount"],
                "dimensions": ["sales.region"],
                "derivedMetrics": [
                    {"name": "avg_price", "expression": "sales.amount/sales.quantity"}
                ],
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        assert "avg_price" in result.data[0]

    anyio.run(run)


@pytest.fixture
def epoch_source(tmp_path):
    csv = tmp_path / "events.csv"
    # Epoch-seconds timestamps (a few distinct values, time-named column).
    csv.write_text(
        "ts,region,amount\n"
        "1704067200,North,100\n"
        "1704153600,South,200\n"
        "1704240000,North,150\n"
        "1704326400,East,300\n"
    )
    s = CsvSource(named_csvs={"events": csv})
    yield s
    s.close()


def test_epoch_time_column_detected(epoch_source):
    profile = profile_dataset(epoch_source)
    semantic = SemanticModel.from_profile(profile)
    assert any(t.ref == "events.ts" for t in semantic.time_columns)
    assert semantic.time_columns[0].encoding.value == "epoch_seconds"


def test_epoch_time_column_trend(epoch_source):
    profile = profile_dataset(epoch_source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(epoch_source, semantic)

    async def run():
        result = await tools.trend().invoke(
            {
                "metrics": ["events.amount"],
                "timeColumn": "events.ts",
                "grain": "day",
                "lastDays": 10000,
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        assert len(result.data) == 4

    anyio.run(run)


@pytest.fixture
def chain_source(tmp_path):
    sales = tmp_path / "sales.csv"
    sales.write_text("region,amount\nNorth,100\nSouth,200\nNorth,50\n")
    regions = tmp_path / "regions.csv"
    regions.write_text("region,manager\nNorth,Alice\nSouth,Bob\n")
    returns = tmp_path / "returns.csv"
    returns.write_text("region,cost\nNorth,10\nSouth,25\n")
    s = CsvSource(named_csvs={"sales": sales, "regions": regions, "returns": returns})
    yield s
    s.close()


def test_multi_hop_join_via_intermediate_table(chain_source):
    # sales & returns connect only through the regions dimension table.
    profile = profile_dataset(chain_source)
    semantic = SemanticModel.from_profile(profile)
    tools = AnalyticsToolset(chain_source, semantic)

    async def run():
        result = await tools.run_query().invoke(
            {
                "metrics": ["sales.amount", "returns.cost"],
                "dimensions": ["regions.manager"],
            },
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content
        by_manager = {r["manager"]: r for r in result.data}
        assert by_manager["Alice"]["amount"] == 150
        assert by_manager["Alice"]["cost"] == 10

    anyio.run(run)


@pytest.fixture
def messy_source(tmp_path):
    # Realistic messy CSV: spaced headers, a quoted numeric with thousands
    # separators, blank rows, and a free-text column.
    csv = tmp_path / "messy.csv"
    csv.write_text(
        '" Order ID ",Region,Amount,Quantity,Date,Notes\n'
        '1,North,"1,000",5,2024-01-01,hello\n'
        '2,South,"2,000",10,2024-01-02,world\n'
        '3,North,"1,500",7,2024-01-03,\n'
        '4,East,"3,000",15,2024-01-04,foo\n'
        "\n"
    )
    s = CsvSource(named_csvs={"messy": csv})
    yield s
    s.close()


def test_messy_csv_profiles_without_error(messy_source):
    profile = profile_dataset(messy_source)
    assert len(profile.tables) == 1
    # Header whitespace / quoting is normalized to 6 columns.
    assert len(profile.columns) == 6
    semantic = SemanticModel.from_profile(profile)
    # A date column and at least one dimension are detected despite the mess.
    assert any(d.column == "Date" for d in semantic.dimensions)
    assert any(d.column == "Region" for d in semantic.dimensions)
    tools = AnalyticsToolset(messy_source, semantic)

    async def run():
        # Quantity survives as a real measure; the comma-formatted Amount is
        # correctly treated as text (a sign of messy input, not a crash).
        result = await tools.run_query().invoke(
            {"metrics": ["messy.Quantity"], "dimensions": ["messy.Region"], "limit": 10},
            RequestContext.ephemeral(),
        )
        assert not result.error, result.content

    anyio.run(run)
