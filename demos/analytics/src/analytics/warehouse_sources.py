"""Warehouse pushdown adapters (P2 scale).

Exposes Postgres / Snowflake / BigQuery as first-class ``DataSource`` backends
behind the ``DataSource`` port, so the analytics engine runs its SQL *on the
warehouse* instead of pulling whole tables into single-node DuckDB. The heavy
lifting is DuckDB's ``ATTACH`` pushdown (already in ``SqlSource``); this module
adds the named, intent-revealing adapters and a factory so a config can say
``warehouse: postgres`` rather than hand-writing attach strings.
"""

from __future__ import annotations

from typing import Any

from demos.analytics.src.analytics.sql_source import SqlSource

# Map a warehouse kind to the DuckDB ATTACH connection string template.
_ATTACH_TEMPLATES: dict[str, str] = {
    "postgres": "postgresql://{uri}",
    "postgresql": "postgresql://{uri}",
    "snowflake": "snowflake://{uri}",
    "bigquery": "bigquery://{uri}",
    "mysql": "mysql://{uri}",
    "sqlite": "sqlite://{uri}",
}


def make_warehouse_source(kind: str, uri: str, alias: str = "wh", db_path: str = ":memory:") -> SqlSource:
    """Create a warehouse-backed ``DataSource``.

    ``kind`` is one of postgres / snowflake / bigquery / mysql / sqlite.
    ``uri`` is the driver connection string (without the scheme). For example::

        make_warehouse_source("postgres", "user:pass@host:5432/db")

    The engine attaches the warehouse read-only; all ``native_query`` SQL is
    pushed down to the warehouse. Raises ``ValueError`` for an unknown kind and
    ``RuntimeError`` if the required DuckDB extension/secret isn't installed.
    """
    template = _ATTACH_TEMPLATES.get(kind.lower())
    if template is None:
        raise ValueError(
            f"unknown warehouse kind '{kind}'; supported: {sorted(_ATTACH_TEMPLATES)}"
        )
    conn_str = template.format(uri=uri)
    try:
        return SqlSource(db_path=db_path, attach={alias: conn_str})
    except Exception as exc:  # pragma: no cover - depends on installed DuckDB extensions
        raise RuntimeError(
            f"could not attach {kind} warehouse (is the DuckDB "
            f"{kind} extension installed?): {exc}"
        ) from exc


def available_kinds() -> list[str]:
    return sorted(_ATTACH_TEMPLATES)
