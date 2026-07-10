"""E2E for the DSL / NL tools wired into ``AnalyticsToolset`` (PR-D4 / PR-D5 hooks).

Verifies an agent can call ``dsl_query`` / ``nl_query`` and get a defensible
answer (rows + planned SQL) without writing raw SQL, and that business synonyms
are honored.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
)
from demos.analytics.src.analytics.toolset import AnalyticsToolset


def _engine_source_model():
    df = pd.DataFrame(
        {
            "region": ["N", "S", "E", "W"] * 3,
            "amount": [10.0, 20.0, 30.0, 40.0] * 3,
        }
    )
    d = tempfile.mkdtemp(prefix="dsl_tools_")
    path = os.path.join(d, "sales.csv")
    df.to_csv(path, index=False)
    src = CsvSource(named_csvs={"sales": __import__("pathlib").Path(path)})
    model = SemanticModel(
        metrics=(Metric(table="sales", column="amount", aggregation="sum"),),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(),
        relationships=(),
    )
    return src, model


@pytest.mark.asyncio
async def test_dsl_query_tool_returns_rows_and_sql():
    src, model = _engine_source_model()
    ts = AnalyticsToolset(src, model)
    tools = {t.spec.name: t for t in ts.all_tools()}
    assert "dsl_query" in tools and "nl_query" in tools

    res = await tools["dsl_query"].invoke(
        {"dsl": "SELECT sales.amount BY sales.region"}, None
    )
    assert res.ok
    assert len(res.data) == 4
    assert res.trust is not None
    src.close()


@pytest.mark.asyncio
async def test_dsl_query_tool_honors_synonyms():
    src, model = _engine_source_model()
    ts = AnalyticsToolset(src, model)
    tools = {t.spec.name: t for t in ts.all_tools()}
    res = await tools["dsl_query"].invoke(
        {"dsl": "SELECT revenue BY region", "synonyms": {"revenue": "sales.amount", "region": "sales.region"}},
        None,
    )
    assert res.ok
    rows = {r["region"]: r["revenue"] for r in res.data}
    assert rows == {"N": 30.0, "S": 60.0, "E": 90.0, "W": 120.0}
    src.close()


@pytest.mark.asyncio
async def test_nl_query_tool_runs_nl_question():
    src, model = _engine_source_model()
    ts = AnalyticsToolset(src, model)
    tools = {t.spec.name: t for t in ts.all_tools()}
    res = await tools["nl_query"].invoke({"question": "show amount by region"}, None)
    assert res.ok
    assert len(res.data) == 4
    # The planned DSL text surfaces in the framed content (machine-readable body).
    assert "show amount by region" in res.content
    src.close()
