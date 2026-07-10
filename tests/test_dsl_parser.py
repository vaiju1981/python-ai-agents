"""PR-D2 verification: textual DSL parser -> query IR.

Covers the gates from docs/dsl_engine.md PR-D2:

1. A full query parses into a ``DslQuery`` with the right fields, and ``.to_spec``
   yields a ``QuerySpec`` whose ``plan()`` produces SQL matching a hand-built
   reference.
2. A calculated metric referenced by name (PR-D1) carries the expanded
   ``derivedMetrics`` and ``plan()`` runs.
3. Malformed input raises ``DslParseError`` naming the offending clause.
4. ``to_text`` round-trips to a normalized form that re-parses.
"""

from __future__ import annotations

import pytest

from demos.analytics.src.analytics.dsl.ast import DslQuery
from demos.analytics.src.analytics.dsl.catalog import (
    CalculatedMetricDef,
    MetricCatalog,
)
from demos.analytics.src.analytics.dsl.parser import DslParseError, parse
from demos.analytics.src.analytics.query_planner import plan
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
)


def _sales_model() -> SemanticModel:
    return SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="sales", column="quantity", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(),
        relationships=(),
    )


def _ref_spec_sql(model: SemanticModel) -> str:
    """Hand-built reference SQL for the main query, computed via the planner."""
    from demos.analytics.src.analytics.query_planner import QuerySpec

    spec = QuerySpec(
        metrics=("sales.amount", "sales.quantity"),
        dimensions=("sales.region",),
        filters=(_ref_filter("sales.amount", ">", "100"),),
        last_days=30,
        order_by="sales.amount",
        descending=True,
        limit=10,
    )
    return plan(model, spec)


def _ref_filter(column, op, value):
    from demos.analytics.src.analytics.query_planner import Filter

    return Filter(column=column, op=op, value=value)


def test_parse_full_query_into_ir():
    q = parse(
        "SELECT sales.amount, sales.quantity BY sales.region "
        "WHERE sales.amount > 100 SINCE 30 DAYS ORDER BY sales.amount DESC LIMIT 10"
    )
    assert isinstance(q, DslQuery)
    assert q.metrics == ("sales.amount", "sales.quantity")
    assert q.dimensions == ("sales.region",)
    assert q.filters[0].column == "sales.amount"
    assert q.filters[0].op == ">"
    assert q.filters[0].value == "100"
    assert q.last_days == 30
    assert q.order_by == "sales.amount"
    assert q.descending is True
    assert q.limit == 10


def test_to_spec_produces_sql_matching_reference():
    model = _sales_model()
    q = parse(
        "SELECT sales.amount, sales.quantity BY sales.region "
        "WHERE sales.amount > 100 SINCE 30 DAYS ORDER BY sales.amount DESC LIMIT 10"
    )
    spec = q.to_spec(model)
    sql = plan(model, spec)
    ref = _ref_spec_sql(model)
    assert sql == ref


def test_calculated_metric_by_name_carries_derived():
    model = _sales_model()
    cat = MetricCatalog(
        [CalculatedMetricDef(name="avg_price", expression="sales.amount/sales.quantity", description="")]
    )
    q = parse("SELECT avg_price BY sales.region")
    spec = q.to_spec(model, cat)
    assert spec.metrics == ()
    assert spec.derivedMetrics == (
        {"name": "avg_price", "expression": "(SUM(\"sales\".\"amount\")/SUM(\"sales\".\"quantity\"))"},
    )
    # plan() must run without error against the expanded spec.
    sql = plan(model, spec)
    assert "avg_price" in sql


def test_inline_expression_requires_alias():
    with pytest.raises(DslParseError):
        parse("SELECT sales.amount / sales.quantity BY sales.region")


def test_inline_expression_with_alias():
    model = _sales_model()
    q = parse("SELECT sales.amount / sales.quantity AS avg_price BY sales.region")
    spec = q.to_spec(model)
    assert spec.derivedMetrics[0]["name"] == "avg_price"


@pytest.mark.parametrize(
    "bad",
    [
        "SELECT BY sales.region",               # missing metric list
        "SELECT sales.amount OP ~ x",            # unknown operator char
        "SELECT sales.amount IN (a,b",           # unbalanced IN list
        "SELECT sales.amount WHERE sales.amount",  # missing comparison after col
        "SELECT sales.amount SINCE 5 HOURS",     # unknown time unit
    ],
)
def test_malformed_raises_scoped_error(bad: str):
    with pytest.raises(DslParseError):
        parse(bad)


def test_roundtrip_to_text_reparses():
    for text in [
        "SELECT sales.amount, sales.quantity BY sales.region WHERE sales.amount > 100 SINCE 30 DAYS ORDER BY sales.amount DESC LIMIT 10",
        "SELECT avg_price BY sales.region WHERE sales.region IN (N, S) AND sales.region LIKE 'N%'",
        "SELECT sales.amount AS total BY sales.region ORDER BY total ASC LIMIT 5",
    ]:
        q = parse(text)
        norm = q.to_text()
        reparsed = parse(norm)
        assert reparsed.to_text() == norm
