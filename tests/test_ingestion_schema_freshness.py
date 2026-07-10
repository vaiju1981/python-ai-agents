"""PR-16 — schema evolution + freshness gate + auto-window (E2E).

Covers:
* Schema evolution: a streamed batch that adds a column is landed via
  ``ALTER TABLE ... ADD COLUMN`` (no full reload); the new column becomes
  queryable and shows up in the rebuilt SemanticModel.
* Freshness gate: with ``ANALYTICS_MAX_STALE_DAYS`` set, a ``dsl_query`` /
  ``run_sql`` against a stale table returns a structured "stale" result rather
  than a silent number; ``allow_stale=true`` overrides.
* Auto-window: a trailing-window query anchors at the table's high-water mark so
  in-window late rows are included even when the data is older than wall-clock now.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
    TimeColumn,
    TimeEncoding,
)
from demos.analytics.src.analytics.toolset import AnalyticsToolset


def _sales_source(tmp_path, rows):
    df = pd.DataFrame(rows)
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    return CsvSource(named_csvs={"sales": path})


def _sales_model():
    return SemanticModel(
        metrics=(Metric(table="sales", column="v", aggregation="sum"),),
        dimensions=(
            Dimension(table="sales", column="id", role="identifier"),
            Dimension(table="sales", column="region"),
        ),
        entity_keys=(),
        time_columns=(TimeColumn(table="sales", column="ts", encoding=TimeEncoding.NATIVE),),
        relationships=(),
    )


def _sales_toolset(tmp_path, rows):
    src = _sales_source(tmp_path, rows)
    ts = AnalyticsToolset(src, _sales_model(), profile=profile_dataset(src))
    return src, ts


def _ok(res) -> bool:
    return not res.error


def _is_stale(res) -> bool:
    return isinstance(res.data, dict) and res.data.get("stale") is True


# --- schema evolution -------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_new_column_evolves_schema_and_model(tmp_path):
    # Base table has id, ts, v. Ingest a batch that also carries a new column.
    src, ts = _sales_toolset(
        tmp_path,
        [
            {"id": 1, "ts": dt.date(2025, 1, 1), "v": 10, "region": "N"},
            {"id": 2, "ts": dt.date(2025, 1, 2), "v": 20, "region": "S"},
        ],
    )
    try:
        new_rows = [
            {"id": 3, "ts": dt.date(2025, 1, 3), "v": 30, "region": "E"},
            {"id": 4, "ts": dt.date(2025, 1, 4), "v": 40, "region": "W"},
            # new 'channel' column arrives mid-stream
            {"id": 5, "ts": dt.date(2025, 1, 5), "v": 50, "region": "N", "channel": "web"},
        ]
        ts.ingest("sales", new_rows, keys=["id"])

        # New column is physically present and queryable.
        cols = [c.name for c in src.tables()[0].columns]
        assert "channel" in cols
        q = src.native_query(
            "SELECT DISTINCT channel FROM sales WHERE channel IS NOT NULL"
        )
        assert {r["channel"] for r in q} == {"web"}

        # Rebuilt model exposes the new column (no full reload happened).
        assert any(d.column == "channel" for d in ts.model.dimensions)

        # dsl_query can group by the new column after refresh.
        tools = {t.spec.name: t for t in ts.all_tools()}
        res = await tools["dsl_query"].invoke(
            {"dsl": "SELECT sales.v BY sales.channel"}, None
        )
        assert _ok(res)
        assert any(r.get("channel") == "web" for r in res.data)
    finally:
        src.close()


@pytest.mark.asyncio
async def test_schema_evolution_partial_batch_does_not_drop_columns(tmp_path):
    src, ts = _sales_toolset(
        tmp_path,
        [{"id": 1, "ts": dt.date(2025, 1, 1), "v": 10, "region": "N"}],
    )
    try:
        # A partial batch carrying a new column but omitting 'region'.
        ts.ingest(
            "sales",
            [{"id": 2, "ts": dt.date(2025, 1, 2), "v": 20, "tier": "gold"}],
            keys=["id"],
        )
        cols = [c.name for c in src.tables()[0].columns]
        assert "region" in cols  # retired-in-batch column still present
        assert "tier" in cols  # new column added
        # The existing row keeps region; the new row is NULL in region, set in tier.
        rows = {r["id"]: r for r in src.native_query("SELECT * FROM sales")}
        assert rows[1]["region"] == "N"
        assert rows[2]["region"] is None
        assert rows[2]["tier"] == "gold"
    finally:
        src.close()


# --- freshness gate ---------------------------------------------------------


@pytest.mark.asyncio
async def test_freshness_gate_blocks_stale_query(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYTICS_MAX_STALE_DAYS", "0")
    # Data is old (2025) relative to now (2026) -> staler than 0 days.
    src, ts = _sales_toolset(
        tmp_path,
        [
            {"id": 1, "ts": dt.date(2025, 1, 1), "v": 10, "region": "N"},
            {"id": 2, "ts": dt.date(2025, 1, 2), "v": 20, "region": "S"},
        ],
    )
    try:
        tools = {t.spec.name: t for t in ts.all_tools()}
        stale = await tools["dsl_query"].invoke(
            {"dsl": "SELECT sales.v BY sales.region"}, None
        )
        assert _ok(stale)
        assert _is_stale(stale)
        assert "sales" in stale.data.get("tables", [])

        # Override answers.
        answered = await tools["dsl_query"].invoke(
            {"dsl": "SELECT sales.v BY sales.region", "allow_stale": True}, None
        )
        assert _ok(answered)
        assert not _is_stale(answered)

        # run_sql respects the same gate.
        stale_sql = await tools["run_sql"].invoke(
            {"sql": 'SELECT SUM("v") AS s FROM "sales"'}, None
        )
        assert _is_stale(stale_sql)
        answered_sql = await tools["run_sql"].invoke(
            {"sql": 'SELECT SUM("v") AS s FROM "sales"', "allow_stale": True}, None
        )
        assert not _is_stale(answered_sql)
    finally:
        src.close()


@pytest.mark.asyncio
async def test_freshness_gate_off_without_policy(tmp_path):
    src, ts = _sales_toolset(
        tmp_path,
        [{"id": 1, "ts": dt.date(2025, 1, 1), "v": 10, "region": "N"}],
    )
    try:
        tools = {t.spec.name: t for t in ts.all_tools()}
        res = await tools["dsl_query"].invoke(
            {"dsl": "SELECT sales.v BY sales.region"}, None
        )
        assert _ok(res)
        assert not _is_stale(res)
    finally:
        src.close()


# --- auto-window ------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_window_includes_in_window_late_rows(tmp_path):
    # Build a table whose max event date is ~120 days in the past, with a late
    # row 1 day before the watermark. With wall-clock anchoring a 30-day window
    # would return nothing; with watermark anchoring it returns both rows.
    today = dt.date.today()
    wm = today - dt.timedelta(days=120)  # high-water mark
    late = wm - dt.timedelta(days=1)  # within 1-day lateness window
    src, ts = _sales_toolset(
        tmp_path,
        [
            {"id": 1, "ts": wm, "v": 100, "region": "N"},
            {"id": 2, "ts": late, "v": 50, "region": "S"},
        ],
    )
    try:
        tools = {t.spec.name: t for t in ts.all_tools()}
        res = await tools["trend"].invoke(
            {
                "metrics": ["sales.v"],
                "timeColumn": "sales.ts",
                "grain": "day",
                "lastDays": 30,
            },
            None,
        )
        assert _ok(res)
        # Both rows are within the watermark-anchored 30-day window.
        assert len(res.data) == 2
        assert sum(r["v"] for r in res.data) == 150
    finally:
        src.close()


@pytest.mark.asyncio
async def test_auto_window_matches_now_anchor_for_fresh_data(tmp_path):
    today = dt.date.today()
    src, ts = _sales_toolset(
        tmp_path,
        [
            {"id": 1, "ts": today - dt.timedelta(days=5), "v": 10, "region": "N"},
            {"id": 2, "ts": today - dt.timedelta(days=40), "v": 20, "region": "S"},
        ],
    )
    try:
        tools = {t.spec.name: t for t in ts.all_tools()}
        res = await tools["trend"].invoke(
            {
                "metrics": ["sales.v"],
                "timeColumn": "sales.ts",
                "grain": "day",
                "lastDays": 30,
            },
            None,
        )
        assert _ok(res)
        # Only the row within 30 days of now is included (watermark == now here).
        assert sum(r["v"] for r in res.data) == 10
    finally:
        src.close()
