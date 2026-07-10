"""PR-D1 verification: calculated-metric catalog (semantic layer).

Covers the gates from docs/dsl_engine.md PR-D1:

1. A calculated metric (e.g. avg_price = amount/quantity) expands to additive-safe
   aggregated SQL so ratios stay correct under GROUP BY.
2. Chained calculated metrics (a calculated metric referencing another) expand
   fully — no unexpanded catalog names leak into the SQL.
3. CASE / NULLIF expressions (safe division) round-trip and stay valid SQL.
4. Cycle detection: two metrics that reference each other raise CatalogError.
5. Validation fails fast on a reference to a non-existent base column.
6. Persistence: JSON round-trip + CatalogStore override merge.
7. dataset_sig-keyed selection: health vs casino yield distinct, tailored
   catalogs with zero engine branching; pure row growth reuses the same catalog;
   a base metric is always resolvable by ref even with no tailored catalog.
"""

from __future__ import annotations

import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.dsl.catalog import (
    CalculatedMetricDef,
    CatalogError,
    CatalogStore,
    MetricCatalog,
    catalog_for_source,
)
from demos.analytics.src.analytics.profiler import profile_dataset
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
            Metric(table="sales", column="cost", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(),
        relationships=(),
    )


def test_resolve_base_metric_by_ref_and_column():
    model = _sales_model()
    cat = MetricCatalog()
    sql, alias = cat.resolve("sales.amount", model)
    assert sql == 'SUM("sales"."amount")'
    assert alias == "amount"
    # Bare column name also resolves to its (unique) base metric.
    sql2, alias2 = cat.resolve("quantity", model)
    assert sql2 == 'SUM("sales"."quantity")'
    assert alias2 == "quantity"


def test_calculated_metric_expands_additive_safe():
    model = _sales_model()
    cat = MetricCatalog(
        [
            CalculatedMetricDef(
                name="avg_price", expression="sales.amount/sales.quantity", description=""
            )
        ]
    )
    sql, alias = cat.resolve("avg_price", model)
    assert alias == "avg_price"
    # Each operand is aggregated, so the ratio is correct under GROUP BY.
    assert sql == '(SUM("sales"."amount")/SUM("sales"."quantity"))'


def test_chained_calculated_metric_expands_fully():
    model = _sales_model()
    cat = MetricCatalog(
        [
            CalculatedMetricDef(
                name="gross", expression="sales.amount - sales.cost", description=""
            ),
            CalculatedMetricDef(name="margin_pct", expression="gross/sales.amount", description=""),
        ]
    )
    sql, _ = cat.resolve("margin_pct", model)
    # No unexpanded catalog name ("gross") may leak into the SQL.
    assert "gross" not in sql
    assert sql.count("SUM(") == 3  # amount, cost, amount again
    assert 'SUM("sales"."amount") - SUM("sales"."cost")' in sql


def test_case_nullif_expression_round_trips():
    model = _sales_model()
    cat = MetricCatalog(
        [
            CalculatedMetricDef(
                name="safe_ratio",
                expression="NULLIF(sales.amount,0)/sales.quantity",
                description="",
            )
        ]
    )
    sql, _ = cat.resolve("safe_ratio", model)
    assert 'NULLIF(SUM("sales"."amount"),0)' in sql
    assert sql.endswith('/SUM("sales"."quantity"))')


def test_cycle_detection_raises():
    model = _sales_model()
    cat = MetricCatalog(
        [
            CalculatedMetricDef(name="a", expression="b", description=""),
            CalculatedMetricDef(name="b", expression="a", description=""),
        ]
    )
    with pytest.raises(CatalogError) as exc:
        cat.resolve("a", model)
    assert "cyclic" in str(exc.value).lower()
    # validate() must also catch it without executing.
    with pytest.raises(CatalogError):
        cat.validate(model)


def test_validation_fails_on_unknown_base_column():
    model = _sales_model()
    cat = MetricCatalog(
        [CalculatedMetricDef(name="bad", expression="sales.nope/sales.quantity", description="")]
    )
    with pytest.raises(CatalogError) as exc:
        cat.validate(model)
    assert "sales.nope" in str(exc.value)


def test_persist_roundtrip_and_override_merge(tmp_path):
    cat = MetricCatalog(
        [
            CalculatedMetricDef(
                name="avg_price", expression="sales.amount/sales.quantity", description=""
            )
        ]
    )
    # JSON round-trip.
    text = cat.to_dict()
    back = MetricCatalog.from_dict(text)
    assert back.names() == cat.names()

    # CatalogStore override merge: dataset-specific metric wins on name clash.
    base = MetricCatalog(
        [CalculatedMetricDef(name="shared", expression="sales.amount", description="")]
    )
    ds = MetricCatalog(
        [CalculatedMetricDef(name="shared", expression="sales.quantity", description="")]
    )
    merged = base.override(ds)
    sql, _ = merged.resolve("shared", _sales_model())
    assert sql == '(SUM("sales"."quantity"))'  # dataset wins

    store = CatalogStore(tmp_path)
    sig = "abc123"
    store.save(sig, cat)
    assert store.path_for(sig).exists()
    assert store.load(sig).names() == ["avg_price"]


def _csv_source(table: str, rows: list[dict[str, float]]) -> CsvSource:
    import os
    import tempfile

    d = tempfile.mkdtemp(prefix="dsl_catalog_")
    path = os.path.join(d, f"{table}.csv")
    header = ",".join(rows[0].keys())
    body = "\n".join(",".join(str(v) for v in r.values()) for r in rows)
    with open(path, "w") as fh:
        fh.write(f"{header}\n{body}\n")
    from pathlib import Path as _P

    return CsvSource(named_csvs={table: _P(path)})


def _health_source(n: int = 10) -> CsvSource:
    # Values cycle on a fixed period (5) so the *distribution* is identical for
    # any n that is a multiple of 5 (pure row growth) -> dataset_sig stays stable.
    return _csv_source(
        "health",
        [{"readmissions": float(i % 5 + 1), "discharges": float(i % 5 + 1)} for i in range(n)],
    )


def _casino_source(n: int = 10) -> CsvSource:
    return _csv_source(
        "casino",
        [{"amount": float(i), "hands": float(i + 2)} for i in range(1, n + 1)],
    )


def test_dataset_sig_keyed_selection_tailors_per_domain(tmp_path):
    health = _health_source()
    casino = _casino_source()
    health_model = SemanticModel.from_profile(profile_dataset(health))
    casino_model = SemanticModel.from_profile(profile_dataset(casino))

    # Tailored calculated metrics, one per domain.
    health_cat = MetricCatalog(
        [
            CalculatedMetricDef(
                name="readmit_rate",
                expression="health.readmissions/health.discharges",
                description="",
            )
        ]
    )
    casino_cat = MetricCatalog(
        [
            CalculatedMetricDef(
                name="avg_bet", expression="casino.amount/casino.hands", description=""
            )
        ]
    )

    store = CatalogStore(tmp_path)
    h_sig = fingerprint(health, row_count_aware=False)
    c_sig = fingerprint(casino, row_count_aware=False)
    assert h_sig != c_sig  # different datasets -> different keys
    store.save(h_sig, health_cat)
    store.save(c_sig, casino_cat)

    # The engine auto-selects the tailored catalog per source.
    h_eff, h_sig2 = catalog_for_source(
        health, model=health_model, catalog_dir=tmp_path, create=False
    )
    c_eff, c_sig2 = catalog_for_source(
        casino, model=casino_model, catalog_dir=tmp_path, create=False
    )
    assert h_sig2 == h_sig and c_sig2 == c_sig

    # Health-only metric resolves for health, not for casino.
    assert "readmit_rate" in h_eff.resolve("readmit_rate", health_model)[1]
    with pytest.raises(CatalogError):
        c_eff.resolve("readmit_rate", casino_model)

    # A base metric is always resolvable by ref, even with no tailored catalog.
    assert h_eff.resolve("health.readmissions", health_model)[0].startswith("SUM(")
    assert c_eff.resolve("casino.amount", casino_model)[0].startswith("SUM(")

    health.close()
    casino.close()


def test_pure_row_growth_reuses_same_catalog(tmp_path):
    small = _health_source(n=10)
    big = _health_source(n=100)  # same columns/distribution -> same dataset_sig

    sig_small = fingerprint(small, row_count_aware=False)
    sig_big = fingerprint(big, row_count_aware=False)
    assert sig_small == sig_big  # pure growth is row-count-agnostic (PR-11)

    store = CatalogStore(tmp_path)
    # Same catalog file is reused across growth.
    assert store.path_for(sig_small) == store.path_for(sig_big)

    small.close()
    big.close()
