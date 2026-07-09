"""PR-5 verification: partial-failure robustness for fan-out joins.

Covers the three E2E gates from docs/production_readiness.md:

1. A deliberately broken/missing table raises a *scoped* ``QueryPlanError``
   naming the offending table (not a generic SQL failure).
2. Best-effort mode drops the broken table and returns the partial result with
   the table listed in ``warnings`` / ``dropped_tables`` — and the partial SQL
   actually executes.
3. Dropping a column post-profiling changes ``dataset_sig`` so a PR-2 cached
   model keyed on the old signature is invalidated (cache miss).
"""

from __future__ import annotations

import pytest

from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.data_source import Relationship
from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.model_store import (
    FileModelStore,
    ModelRecord,
    model_key,
)
from demos.analytics.src.analytics.query_planner import (
    MissingTableError,
    QueryPlanError,
    QuerySpec,
    SchemaContractError,
    plan_query,
)
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
)


def _two_fact_source(tmp_path):
    sales = tmp_path / "sales.csv"
    sales.write_text("region,amount\nNorth,100\nSouth,200\nNorth,50\n")
    returns = tmp_path / "returns.csv"
    returns.write_text("region,cost\nNorth,10\nSouth,25\n")
    s = CsvSource(named_csvs={"sales": sales, "returns": returns})
    return s


def _model_with_ghost() -> SemanticModel:
    """A model that references a real table plus a non-existent 'ghost' table."""
    return SemanticModel(
        metrics=(
            Metric(table="sales", column="amount", aggregation="sum"),
            Metric(table="ghost", column="cost", aggregation="sum"),
        ),
        dimensions=(Dimension(table="sales", column="region"),),
        entity_keys=(),
        time_columns=(),
        relationships=(
            Relationship(
                from_table="sales", from_columns=("region",),
                to_table="returns", to_columns=("region",),
            ),
        ),
    )


def test_missing_table_raises_scoped_error(tmp_path):
    src = _two_fact_source(tmp_path)
    model = _model_with_ghost()
    with pytest.raises(MissingTableError) as exc:
        plan_query(model, QuerySpec(metrics=("sales.amount", "ghost.cost")), src)
    assert exc.value.table == "ghost"
    assert isinstance(exc.value, QueryPlanError)
    src.close()


def test_best_effort_drops_broken_table_and_runs(tmp_path):
    src = _two_fact_source(tmp_path)
    model = _model_with_ghost()
    res = plan_query(
        model, QuerySpec(metrics=("sales.amount", "ghost.cost")), src, best_effort=True
    )
    assert res.dropped_tables == ["ghost"]
    assert res.warnings
    assert "ghost" not in res.sql
    assert "sales" in res.sql
    # The partial query must actually execute against the source.
    rows = src.native_query_with_limit(res.sql, 10)
    assert rows and "amount" in rows[0]
    src.close()


def test_schema_contract_breaks_on_missing_column(tmp_path):
    src = _two_fact_source(tmp_path)
    model = SemanticModel(
        metrics=(Metric(table="sales", column="amountx", aggregation="sum"),),
        dimensions=(),
        entity_keys=(),
        time_columns=(),
        relationships=(
            Relationship(
                from_table="sales", from_columns=("region",),
                to_table="returns", to_columns=("region",),
            ),
        ),
    )
    with pytest.raises(SchemaContractError) as exc:
        plan_query(model, QuerySpec(metrics=("sales.amountx",)), src)
    assert exc.value.table == "sales"
    src.close()


def test_dropped_column_invalidates_model_cache(tmp_path):
    full = tmp_path / "full.csv"
    full.write_text("region,amount\nNorth,100\nSouth,200\n")
    src_full = CsvSource(named_csvs={"sales": full})
    sig_full = fingerprint(src_full)

    # Same data but the 'amount' column has been dropped.
    slim = tmp_path / "slim.csv"
    slim.write_text("region\nNorth\nSouth\n")
    src_slim = CsvSource(named_csvs={"sales": slim})
    sig_slim = fingerprint(src_slim)

    assert sig_full != sig_slim

    store = FileModelStore(directory=tmp_path / "models")
    key_full = model_key(
        dataset_sig=sig_full, task="t", target="y", predictors=["x"], algorithm="linear"
    )
    key_slim = model_key(
        dataset_sig=sig_slim, task="t", target="y", predictors=["x"], algorithm="linear"
    )
    store.put(ModelRecord(key=key_full, model={"v": 1}, metadata={}, trained_at=0.0))

    # Cached under the old signature -> hit; new signature -> miss (invalidated).
    assert store.get(key_full) is not None
    assert store.get(key_slim) is None

    src_full.close()
    src_slim.close()
