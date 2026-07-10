"""PR-14 — incremental model / semantic refresh wired to ingest (E2E).

Verifies that ingesting new data through AnalyticsToolset keeps the live
SemanticModel and the cached DSL engines in sync, so dsl_query / nl_query
reflect arriving data without a full rebuild.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
)
from demos.analytics.src.analytics.toolset import AnalyticsToolset


def _source_model():
    df = pd.DataFrame(
        {
            "region": ["N", "S", "E", "W"] * 3,
            "amount": [10.0, 20.0, 30.0, 40.0] * 3,
        }
    )
    d = tempfile.mkdtemp(prefix="pr14_")
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
async def test_ingest_refreshes_dsl_results():
    src, model = _source_model()
    ts = AnalyticsToolset(src, model, profile=profile_dataset(src))
    try:
        tools = {t.spec.name: t for t in ts.all_tools()}
        before = await tools["dsl_query"].invoke(
            {"dsl": "SELECT sales.amount BY sales.region"}, None
        )
        assert before.ok
        baseline_total = sum(r["amount"] for r in before.data)
        assert baseline_total == 300.0

        # Ingest a new region row; the model + cached DSL engine must refresh.
        result = ts.ingest("sales", [{"region": "Z", "amount": 500.0}], keys=["region"])
        assert result.inserted == 1
        # Cache was cleared by refresh; a subsequent query rebuilds it.
        assert ts._dsl_engine_cache == {}

        after = await tools["dsl_query"].invoke(
            {"dsl": "SELECT sales.amount BY sales.region"}, None
        )
        assert after.ok
        rows = {r["region"]: r["amount"] for r in after.data}
        assert rows["Z"] == 500.0
        assert sum(rows.values()) == 800.0  # baseline 300 + new 500
        # Engine cache repopulated after the query.
        assert ts._dsl_engine_cache
    finally:
        src.close()


@pytest.mark.asyncio
async def test_refresh_after_ingest_rebuilds_model_and_clears_cache():
    src, model = _source_model()
    ts = AnalyticsToolset(src, model, profile=profile_dataset(src))
    try:
        sig = ts.refresh_after_ingest(
            delta_rows={"sales": [{"region": "Z", "amount": 500.0}]}
        )
        assert isinstance(sig, str) and len(sig) == 16
        # Model rebuilt from the updated profile; metric structure preserved.
        assert any(m.column == "amount" for m in ts.model.metrics)
        assert ts._dsl_engine_cache == {}
    finally:
        src.close()


@pytest.mark.asyncio
async def test_refresh_without_delta_reprofiles_source():
    src, model = _source_model()
    ts = AnalyticsToolset(src, model)  # no profile provided -> built lazily
    try:
        sig = ts.refresh_after_ingest()  # refresh path: re-profile from source
        assert isinstance(sig, str) and len(sig) == 16
        assert any(m.column == "amount" for m in ts.model.metrics)
    finally:
        src.close()
