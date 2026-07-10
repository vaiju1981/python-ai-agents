"""PR-D3 verification: name grounding / synonym resolver.

Covers the gates from docs/dsl_engine.md PR-D3:

1. A query written in business vocabulary grounds to the literal-ref query and
   produces identical SQL: ``revenue`` -> ``sales.amount``, ``region`` ->
   ``sales.region``, ``north`` -> ``N``, ``30 days`` -> ``last_days=30``.
2. Synonym overrides take precedence and are case/whitespace insensitive
   (``net win`` -> ``sales.net_win``).
3. Unknown tokens raise ``UnresolvedNameError`` naming the kind.
4. Fuzzy fallback (flag on) maps ``revenu`` -> ``revenue`` within a distance
   budget; with fuzzy off it raises.
5. Period phrases normalize via ``resolve_period``.
"""

from __future__ import annotations

import pytest

from demos.analytics.src.analytics.dsl.catalog import BaseMetricDef, MetricCatalog
from demos.analytics.src.analytics.dsl.grounding import (
    NameResolver,
    UnresolvedNameError,
)
from demos.analytics.src.analytics.dsl.parser import parse
from demos.analytics.src.analytics.query_planner import plan
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
    TimeColumn,
)


def _sales_model() -> SemanticModel:
    return SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="sales", column="quantity", aggregation="sum"),
            Metric(table="sales", column="net_win", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(TimeColumn(table="sales", column="day", encoding="date"),),
        relationships=(),
    )


def _resolver() -> NameResolver:
    return NameResolver(
        synonyms={"revenue": "sales.amount", "region": "sales.region"},
        value_synonyms={"sales.region": {"north": "N", "south": "S"}},
        period_synonyms={"last 30 days": "30 days", "last month": "30 days"},
    )


def test_grounded_query_matches_literal_ref_sql():
    model = _sales_model()
    cat = MetricCatalog()
    resolver = _resolver()

    grounded = parse(
        "SELECT revenue BY region WHERE region = north SINCE 30 days"
    ).to_spec(model, cat, resolver)
    literal = parse(
        "SELECT sales.amount BY sales.region WHERE sales.region = N SINCE 30 days"
    ).to_spec(model, cat)

    assert plan(model, grounded) == plan(model, literal)
    # The filter value was normalized to the coded value.
    assert grounded.filters[0].value == "N"
    assert grounded.last_days == 30


def test_synonym_override_precedence_and_case_insensitive():
    model = _sales_model()
    cat = MetricCatalog()
    resolver = NameResolver(synonyms={"NET WIN": "sales.net_win"})

    # Grounding precedence at the resolver level (multi-word names are quoted in
    # DSL text, but the resolver itself is fed the bare friendly name).
    defn = resolver.resolve_metric("net win", cat, model)
    assert isinstance(defn, BaseMetricDef)
    assert f"{defn.table}.{defn.column}" == "sales.net_win"

    # Quoted in the textual DSL so the parser keeps the multi-word name intact.
    spec = parse('SELECT "net win" BY sales.region').to_spec(model, cat, resolver)
    assert spec.metrics == ("sales.net_win",)

    # Case/whitespace insensitivity.
    defn2 = resolver.resolve_metric("  NeT  WiN  ", cat, model)
    assert f"{defn2.table}.{defn2.column}" == "sales.net_win"


def test_unknown_token_raises_scoped_error():
    model = _sales_model()
    cat = MetricCatalog()
    resolver = _resolver()

    with pytest.raises(UnresolvedNameError) as exc:
        resolver.resolve_metric("nonexistent", cat, model)
    assert exc.value.kind == "metric"

    with pytest.raises(UnresolvedNameError) as exc2:
        resolver.resolve_dimension("bogus", model)
    assert exc2.value.kind == "dimension"


def test_fuzzy_fallback_maps_within_budget():
    model = _sales_model()
    cat = MetricCatalog()
    resolver = NameResolver(synonyms={"revenue": "sales.amount"}, fuzzy=True, fuzzy_budget=2)

    spec = parse("SELECT revenu BY sales.region").to_spec(model, cat, resolver)
    assert spec.metrics == ("sales.amount",)


def test_fuzzy_off_raises():
    model = _sales_model()
    cat = MetricCatalog()
    resolver = NameResolver(synonyms={"revenue": "sales.amount"}, fuzzy=False)
    with pytest.raises(UnresolvedNameError):
        resolver.resolve_metric("revenu", cat, model)


def test_resolve_period_normalizes_phrases():
    resolver = _resolver()
    assert resolver.resolve_period("last 30 days") == "30 days"
    assert resolver.resolve_period("last month") == "30 days"
    # Unknown phrase passes through unchanged (parser handles it directly).
    assert resolver.resolve_period("since 2024-01-01") == "since 2024-01-01"


def test_callable_value_normalizer():
    # value_synonyms values may be callables (DIMENSION_VALUES_DETECTION_MAPPER
    # style normalizers), not just static strings.
    resolver = NameResolver(
        synonyms={"region": "sales.region"},
        value_synonyms={
            "sales.region": {"north": "N", "south": lambda v: v.upper()[:1]}
        },
    )
    assert resolver.resolve_value("sales.region", "north") == "N"
    assert resolver.resolve_value("sales.region", "south") == "S"
    # Unknown value passes through unchanged.
    assert resolver.resolve_value("sales.region", "east") == "east"
