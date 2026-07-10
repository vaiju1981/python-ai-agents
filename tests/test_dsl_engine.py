"""PR-D4 verification: DSL engine orchestrator + planner integration.

Covers the gates from docs/dsl_engine.md PR-D4:

1. End-to-end over a CSV source: a business-vocabulary query returns aggregate
   rows equal to a hand-computed pandas baseline on the same data.
2. A calculated metric referenced by name executes and matches the inline-
   expression result.
3. A multi-table query joins via discovered relationships (fan-out-safe) and
   matches the baseline.
4. ``best_effort=True`` with a dropped table yields ``warnings`` and partial rows.
5. ``explain`` returns valid SQL that, run directly, yields the same rows.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.dsl.catalog import CalculatedMetricDef, MetricCatalog
from demos.analytics.src.analytics.dsl.engine import DslEngine
from demos.analytics.src.analytics.dsl.grounding import NameResolver
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    Relationship,
    SemanticModel,
    TimeColumn,
)


def _csv_source(table: str, df: pd.DataFrame) -> CsvSource:
    d = tempfile.mkdtemp(prefix="dsl_engine_")
    path = os.path.join(d, f"{table}.csv")
    df.to_csv(path, index=False)
    return CsvSource(named_csvs={table: Path(path)})


def _sales_df(n: int = 12) -> pd.DataFrame:
    today = date.today()
    rows = []
    for i in range(n):
        region = ["N", "S", "E", "W"][i % 4]
        rows.append(
            {
                "region": region,
                "amount": float((i + 1) * 10),
                "quantity": float((i % 3) + 1),
                "day": (today - timedelta(days=i)).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _sales_model() -> SemanticModel:
    return SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="sales", column="quantity", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(TimeColumn(table="sales", column="day", encoding="date"),),
        relationships=(),
    )


def test_end_to_end_against_pandas_baseline():
    df = _sales_df()
    src = _csv_source("sales", df)
    model = _sales_model()
    engine = DslEngine(
        src, model, synonyms={"revenue": "sales.amount"}, best_effort=False
    )

    result = engine.query(
        "SELECT revenue BY region SINCE 30 days ORDER BY revenue DESC LIMIT 5"
    )
    assert result.warnings == []
    assert result.rows  # non-empty

    # Baseline: group by region, sum amount, within the last 30 days.
    baseline = df.groupby("region")["amount"].sum().sort_values(ascending=False)
    got = {r["region"]: r["revenue"] for r in result.rows}
    for region, total in baseline.items():
        assert got[region] == pytest.approx(total)

    # explain() SQL, run directly, yields the same aggregate values.
    sql = engine.explain(
        "SELECT revenue BY region SINCE 30 days ORDER BY revenue DESC LIMIT 5"
    )
    direct = src.native_query(sql)
    direct_map = {r["region"]: r["amount"] for r in direct}
    assert direct_map == got
    src.close()


def test_calculated_metric_by_name_matches_inline():
    df = _sales_df()
    src = _csv_source("sales", df)
    model = _sales_model()
    catalog = MetricCatalog(
        [CalculatedMetricDef(name="avg_price", expression="sales.amount/sales.quantity", description="")]
    )
    engine = DslEngine(src, model, catalog=catalog)

    by_name = engine.query("SELECT avg_price BY sales.region")
    inline = engine.query("SELECT sales.amount / sales.quantity AS avg_price BY sales.region")

    name_rows = {r["region"]: r["avg_price"] for r in by_name.rows}
    inline_rows = {r["region"]: r["avg_price"] for r in inline.rows}
    assert name_rows == inline_rows

    # Baseline: additive-safe avg price = sum(amount)/sum(quantity) per region.
    base = (
        df.groupby("region")
        .apply(lambda g: g["amount"].sum() / g["quantity"].sum(), include_groups=False)
        .to_dict()
    )
    for region, val in base.items():
        assert name_rows[region] == pytest.approx(val)
    src.close()


def test_multi_table_join_matches_baseline():
    sales = _sales_df(n=12)
    regions = pd.DataFrame(
        {"region": ["N", "S", "E", "W"], "manager": ["al", "bo", "cy", "di"]}
    )
    d = tempfile.mkdtemp(prefix="dsl_engine_join_")
    s_path = os.path.join(d, "sales.csv")
    r_path = os.path.join(d, "regions.csv")
    sales.to_csv(s_path, index=False)
    regions.to_csv(r_path, index=False)
    src = CsvSource(named_csvs={"sales": Path(s_path), "regions": Path(r_path)})

    model = SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="sales", column="quantity", aggregation="sum"),
        ),
        dimensions=(
            Dimension(table="sales", column="region"),
            Dimension(table="regions", column="manager"),
        ),
        entity_keys=(),
        time_columns=(),
        relationships=(
            Relationship(
                from_table="sales",
                from_columns=("region",),
                to_table="regions",
                to_columns=("region",),
                cardinality="many_to_one",
            ),
        ),
    )
    engine = DslEngine(src, model)

    result = engine.query("SELECT sales.amount BY regions.manager")
    got = {r["manager"]: r["amount"] for r in result.rows}

    merged = sales.merge(regions, on="region")
    base = merged.groupby("manager")["amount"].sum().to_dict()
    for manager, total in base.items():
        assert got[manager] == pytest.approx(total)
    src.close()


def test_best_effort_drops_missing_table_with_warning():
    df = _sales_df()
    src = _csv_source("sales", df)
    # Model references a table ("phantom") that is absent from the source.
    model = SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="phantom", column="x", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(),
        relationships=(),
    )

    strict = DslEngine(src, model, best_effort=False)
    with pytest.raises(Exception):
        strict.query("SELECT sales.amount, phantom.x BY sales.region")

    engine = DslEngine(src, model, best_effort=True)
    result = engine.query("SELECT sales.amount, phantom.x BY sales.region")
    assert result.warnings  # phantom was dropped
    assert result.rows
    # The surviving metric still aggregates correctly.
    base = df.groupby("region")["amount"].sum().to_dict()
    got = {r["region"]: r["amount"] for r in result.rows}
    for region, total in base.items():
        assert got[region] == pytest.approx(total)
    src.close()
