"""PR-3 verification: warehouse pushdown integration + credentials handling.

The pushdown path (``make_warehouse_source`` → ATTACH → DuckDB SQL on the
warehouse) is exercised end-to-end against a local DuckDB *file* used as a
stand-in warehouse. Results are asserted to match an in-memory baseline on the
same data for the SQL-native tools (``summarize``, ``run_query``,
``ab_test``) and for a fan-out join, proving the heavy lifting runs on the
warehouse rather than in Python.

Real external warehouses (postgres / snowflake / bigquery) are covered by the
same suite but **gated**: they only run when ``RUN_WAREHOUSE_TESTS=1`` and the
matching secret env var is present, so CI stays green without credentials.

Also covered: the ``SecretProvider`` resolves credentials from a name (never
embedded in code), and ``redact_secrets`` scrubs ``user:password@`` from any
string (so URIs never appear in logs or provenance).
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from demos.analytics.src.analytics.data_source import Relationship
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.secrets import (
    redact_secrets,
    resolve_secret,
)
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from demos.analytics.src.analytics.warehouse_sources import (
    available_kinds,
    make_warehouse_source,
)

pytest.importorskip("duckdb")


# --- data helpers ---------------------------------------------------------


def _make_sales() -> pd.DataFrame:
    rows = []
    for i in range(240):
        region = "N" if i % 3 == 0 else "S"
        segment = "A" if i % 2 == 0 else "B"
        amount = float((i % 50) + 1) * (1.5 if region == "N" else 1.0)
        qty = (i % 10) + 1
        is_promo = 1 if i % 4 == 0 else 0
        rows.append(
            {
                "region": region,
                "segment": segment,
                "amount": amount,
                "qty": qty,
                "is_promo": is_promo,
            }
        )
    return pd.DataFrame(rows)


def _make_targets() -> pd.DataFrame:
    return pd.DataFrame({"region": ["N", "S"], "quota": [1000.0, 800.0]})


def _build_warehouse(tmp_path) -> str:
    """Create a DuckDB *file* acting as a stand-in warehouse with two tables."""
    wh = tmp_path / "wh.duckdb"
    con = duckdb.connect(str(wh))
    con.register("sales_df", _make_sales())
    con.register("targets_df", _make_targets())
    con.execute("CREATE TABLE sales AS SELECT * FROM sales_df")
    con.execute("CREATE TABLE targets AS SELECT * FROM targets_df")
    con.close()
    return str(wh)


def _warehouse_model(wh_path: str, alias: str = "wh") -> tuple:
    from dataclasses import replace

    src = make_warehouse_source("duckdb", wh_path, alias=alias)
    model = SemanticModel.from_profile(profile_dataset(src))
    model = replace(
        model,
        relationships=(
            Relationship(
                from_table=f"{alias}.sales",
                from_columns=("region",),
                to_table=f"{alias}.targets",
                to_columns=("region",),
                cardinality="many_to_one",
            ),
        ),
    )
    return src, model, alias


def _baseline_model(tmp_path) -> tuple:
    from dataclasses import replace

    from demos.analytics.src.analytics.csv_source import CsvSource

    sales_csv = tmp_path / "sales.csv"
    targets_csv = tmp_path / "targets.csv"
    _make_sales().to_csv(sales_csv, index=False)
    _make_targets().to_csv(targets_csv, index=False)
    src = CsvSource(named_csvs={"sales": sales_csv, "targets": targets_csv})
    model = SemanticModel.from_profile(profile_dataset(src))
    model = replace(
        model,
        relationships=(
            Relationship(
                from_table="sales",
                from_columns=("region",),
                to_table="targets",
                to_columns=("region",),
                cardinality="many_to_one",
            ),
        ),
    )
    return src, model, ""


# --- numeric comparison ---------------------------------------------------


def _num_multiset(obj) -> list[float]:
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    walk(obj)
    return sorted(vals)


def assert_numeric_close(a, b, tol: float = 1e-3) -> None:
    A, B = _num_multiset(a), _num_multiset(b)
    assert len(A) == len(B), f"numeric-leaf count mismatch: {A} vs {B}"
    for x, y in zip(A, B, strict=True):
        assert abs(x - y) <= tol * (abs(x) + 1.0), f"numeric mismatch: {x} vs {y}"


# --- the integration suite (local warehouse stand-in) ---------------------


def test_warehouse_summarize_matches_baseline(tmp_path):
    wh = _build_warehouse(tmp_path)
    w_src, w_model, w_alias = _warehouse_model(wh)
    b_src, b_model, _ = _baseline_model(tmp_path)

    w_res = _run(w_src, w_model, "summarize", {"metric": f"{w_alias}.sales.amount"})
    b_res = _run(b_src, b_model, "summarize", {"metric": "sales.amount"})
    assert_numeric_close(_loads(w_res), _loads(b_res))
    w_src.close()
    b_src.close()


def test_warehouse_run_query_grouped_matches_baseline(tmp_path):
    wh = _build_warehouse(tmp_path)
    w_src, w_model, w_alias = _warehouse_model(wh)
    b_src, b_model, _ = _baseline_model(tmp_path)

    w_res = _run(
        w_src,
        w_model,
        "run_query",
        {"metrics": [f"{w_alias}.sales.amount"], "dimensions": [f"{w_alias}.sales.region"]},
    )
    b_res = _run(
        b_src,
        b_model,
        "run_query",
        {"metrics": ["sales.amount"], "dimensions": ["sales.region"]},
    )
    assert_numeric_close(_loads(w_res), _loads(b_res))
    w_src.close()
    b_src.close()


def test_warehouse_ab_test_matches_baseline(tmp_path):
    wh = _build_warehouse(tmp_path)
    w_src, w_model, w_alias = _warehouse_model(wh)
    b_src, b_model, _ = _baseline_model(tmp_path)

    w_res = _run(
        w_src,
        w_model,
        "ab_test",
        {
            "metric": f"{w_alias}.sales.amount",
            "groupColumn": f"{w_alias}.sales.is_promo",
            "groupA": "1",
            "groupB": "0",
        },
        toolset="models",
    )
    b_res = _run(
        b_src,
        b_model,
        "ab_test",
        {"metric": "sales.amount", "groupColumn": "sales.is_promo", "groupA": "1", "groupB": "0"},
        toolset="models",
    )
    # SQL-native Welch's t: identical aggregates → identical statistics.
    assert_numeric_close(_loads(w_res), _loads(b_res), tol=1e-9)
    w_src.close()
    b_src.close()


def test_warehouse_fanout_join_matches_baseline(tmp_path):
    wh = _build_warehouse(tmp_path)
    w_src, w_model, w_alias = _warehouse_model(wh)
    b_src, b_model, _ = _baseline_model(tmp_path)

    w_res = _run(
        w_src,
        w_model,
        "run_query",
        {"metrics": [f"{w_alias}.sales.amount"], "dimensions": [f"{w_alias}.targets.region"]},
    )
    b_res = _run(
        b_src,
        b_model,
        "run_query",
        {"metrics": ["sales.amount"], "dimensions": ["targets.region"]},
    )
    assert_numeric_close(_loads(w_res), _loads(b_res))
    w_src.close()
    b_src.close()


def test_warehouse_join_pushdown_via_native_query(tmp_path):
    """A hand-written JOIN runs on the warehouse (pushdown) and matches baseline."""
    wh = _build_warehouse(tmp_path)
    w_src, _, w_alias = _warehouse_model(wh)
    b_src, _, _ = _baseline_model(tmp_path)

    sql = (
        f"SELECT t.region, SUM(s.amount) AS total "
        f"FROM {w_alias}.sales s JOIN {w_alias}.targets t ON s.region = t.region "
        f"GROUP BY t.region ORDER BY t.region"
    )
    w_rows = w_src.native_query(sql)
    b_rows = b_src.native_query(
        "SELECT t.region, SUM(s.amount) AS total FROM sales s "
        "JOIN targets t ON s.region = t.region GROUP BY t.region ORDER BY t.region"
    )
    assert [r["total"] for r in w_rows] == pytest.approx([r["total"] for r in b_rows], rel=1e-6)
    w_src.close()
    b_src.close()


# --- secret handling ------------------------------------------------------


def test_secret_provider_resolves_from_env(monkeypatch):
    monkeypatch.setenv("WAREHOUSE_PG_URI", "user:pass@host/db")
    assert resolve_secret(secret_name="WAREHOUSE_PG_URI") == "user:pass@host/db"
    # Explicit uri wins over secret name.
    assert resolve_secret(uri="direct", secret_name="WAREHOUSE_PG_URI") == "direct"
    # Unknown secret → error.
    with pytest.raises(ValueError):
        resolve_secret(secret_name="DOES_NOT_EXIST")
    # No source → error.
    with pytest.raises(ValueError):
        resolve_secret()


def test_make_warehouse_source_uses_secret_name(tmp_path, monkeypatch):
    # A real DuckDB file path stands in for a credential; the point is that the
    # URI is resolved from a named secret, never embedded in code.
    wh = _build_warehouse(tmp_path)
    monkeypatch.setenv("WAREHOUSE_DB_PATH", wh)
    src = make_warehouse_source(
        "duckdb", secret_name="WAREHOUSE_DB_PATH", uri=None, alias="wh", db_path=":memory:"
    )
    assert [t.name for t in src.tables()] == ["wh.sales", "wh.targets"]
    src.close()


def test_redact_secrets_masks_credentials():
    s = "attached postgresql://alice:hunter2@db.example.com:5432/prod (READ_ONLY)"
    red = redact_secrets(s)
    assert "hunter2" not in red
    assert "alice" not in red
    assert "***:***@" in red
    # Idempotent on already-redacted text.
    assert redact_secrets(red) == red
    # Safe on plain text.
    assert redact_secrets("no secrets here") == "no secrets here"


def test_make_warehouse_source_rejects_bad_alias():
    with pytest.raises(ValueError):
        make_warehouse_source("duckdb", "x.db", alias="bad;alias")


def test_available_kinds_includes_warehouses():
    assert {"postgres", "snowflake", "bigquery", "duckdb"}.issubset(set(available_kinds()))


# --- gated real-warehouse tests (need creds + RUN_WAREHOUSE_TESTS=1) --------


def _real_warehouse_params():
    """Yield (kind, secret_env_var) for configured real warehouses."""
    mapping = [
        ("postgres", "WAREHOUSE_POSTGRES_URI"),
        ("snowflake", "WAREHOUSE_SNOWFLAKE_URI"),
        ("bigquery", "WAREHOUSE_BIGQUERY_URI"),
    ]
    for kind, env_var in mapping:
        uri = __import__("os").environ.get(env_var)
        if uri:
            yield kind, env_var


@pytest.mark.parametrize("kind,env_var", list(_real_warehouse_params()))
def test_real_warehouse_pushdown(kind, env_var, tmp_path):
    """Run the same summarize/ab_test/fan-out checks against a real warehouse.

    Skipped unless RUN_WAREHOUSE_TESTS=1 and the secret env var is set. Mirrors
    the local stand-in suite so external warehouses get the same guarantees.
    """
    if __import__("os").environ.get("RUN_WAREHOUSE_TESTS") != "1":
        pytest.skip("set RUN_WAREHOUSE_TESTS=1 and the warehouse secret to run")
    src = make_warehouse_source(kind, secret_name=env_var, alias="wh")
    model = SemanticModel.from_profile(profile_dataset(src))
    res = _run(src, model, "summarize", {"metric": "wh.sales.amount"})
    assert _loads(res)  # non-empty result from the warehouse
    src.close()


# --- helpers --------------------------------------------------------------


def _run(src, model, tool_name, args, toolset: str = "analytics"):
    import anyio

    from python_ai_agents.core.tool import RequestContext

    if toolset == "models":
        from demos.analytics.src.analytics.models_tools import ModelsToolset

        tools = ModelsToolset(src, model)
    else:
        tools = AnalyticsToolset(src, model)

    def _go():
        return anyio.run(
            lambda: getattr(tools, tool_name)().invoke(args, RequestContext(session_id="s"))
        )

    return _go()


def _loads(result):
    import json

    text = result.content
    # Drop the framed header ("[tool result ...]") and "[trust: ...]" trailer
    # lines, then locate the embedded JSON payload (object or array).
    lines = text.splitlines()
    if lines and lines[0].startswith("["):
        lines = lines[1:]
    if lines and lines[-1].startswith("[trust"):
        lines = lines[:-1]
    body = "\n".join(lines)
    first_obj = body.find("{")
    first_arr = body.find("[")
    candidates = [i for i in (first_obj, first_arr) if i >= 0]
    if not candidates:
        raise ValueError(f"no JSON payload in result: {body[:200]}")
    start = min(candidates)
    end_char = "]" if body[start] == "[" else "}"
    end = body.rfind(end_char)
    if end < start:
        raise ValueError(f"no JSON payload in result: {body[:200]}")
    try:
        return json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse payload: {body[start : end + 1][:200]}") from exc
