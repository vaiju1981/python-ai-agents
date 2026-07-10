"""Warehouse pushdown adapters (P2 scale).

Exposes Postgres / Snowflake / BigQuery / MySQL / SQLite / DuckDB-file as
first-class ``DataSource`` backends behind the ``DataSource`` port, so the
analytics engine runs its SQL *on the warehouse* instead of pulling whole
tables into single-node DuckDB.

Credentials are supplied via a ``SecretProvider`` (``secrets.EnvSecretProvider``
by default) and never embedded in code or config. The connection URI is resolved
from a *secret name* at runtime and is never logged or written into provenance.
"""

from __future__ import annotations

import re
from typing import Any

from demos.analytics.src.analytics.data_source import sql_qcol
from demos.analytics.src.analytics.secrets import (
    EnvSecretProvider,
    SecretProvider,
    redact_secrets,
    resolve_secret,
)
from demos.analytics.src.analytics.sql_source import SqlSource

# Map a warehouse kind to the DuckDB ATTACH connection-string template.
_ATTACH_TEMPLATES: dict[str, str] = {
    "postgres": "postgresql://{uri}",
    "postgresql": "postgresql://{uri}",
    "snowflake": "snowflake://{uri}",
    "bigquery": "bigquery://{uri}",
    "mysql": "mysql://{uri}",
    "sqlite": "sqlite://{uri}",
    "duckdb": "{uri}",
    "duckdb_file": "{uri}",
}

# Aliases become raw SQL identifiers in ATTACH; restrict to a safe shape.
_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def make_warehouse_source(
    kind: str,
    uri: str | None = None,
    *,
    secret_name: str | None = None,
    secret_provider: SecretProvider | None = None,
    alias: str = "wh",
    db_path: str = ":memory:",
) -> SqlSource:
    """Create a warehouse-backed ``DataSource``.

    ``kind`` is one of postgres / snowflake / bigquery / mysql / sqlite /
    duckdb / duckdb_file. For real warehouses, pass ``secret_name`` (resolved
    via ``secret_provider``, default ``EnvSecretProvider``) instead of a literal
    ``uri`` so credentials never live in code/config.

    Example::

        make_warehouse_source("postgres", secret_name="WAREHOUSE_POSTGRES_URI")

    The engine attaches the warehouse read-only; all ``native_query`` SQL is
    pushed down to the warehouse. Raises ``ValueError`` for an unknown kind or a
    bad alias, and ``RuntimeError`` if the required DuckDB extension/secret
    isn't installed.
    """
    if not _ALIAS_RE.match(alias):
        raise ValueError(
            f"alias '{alias}' is not a safe SQL identifier "
            "(use letters, digits, underscores; must start with a letter)"
        )
    template = _ATTACH_TEMPLATES.get(kind.lower())
    if template is None:
        raise ValueError(f"unknown warehouse kind '{kind}'; supported: {sorted(_ATTACH_TEMPLATES)}")
    resolved = resolve_secret(
        uri=uri, secret_name=secret_name, provider=secret_provider or EnvSecretProvider()
    )
    conn_str = template.format(uri=resolved)
    try:
        src = SqlSource(db_path=db_path, attach={alias: conn_str})
        # Mark so the serving layer only pushes scoring to an actual warehouse
        # (PR-8) and never rewrites local/CSV predict behavior.
        src._is_warehouse = True  # type: ignore[attr-defined]
        return src
    except Exception as exc:  # pragma: no cover - depends on installed DuckDB extensions
        raise RuntimeError(
            f"could not attach {kind} warehouse (is the DuckDB "
            f"{kind} extension installed?): {redact_secrets(str(exc))}"
        ) from exc


def available_kinds() -> list[str]:
    return sorted(_ATTACH_TEMPLATES)


def score_warehouse(
    source: Any,
    frame_sql: str,
    model: Any,
    feature_cols: list[str],
    *,
    task: str | None = None,
) -> str | None:
    """Emit in-warehouse SQL that scores ``frame_sql`` with a trained ``model``.

    Returns a SQL ``SELECT`` computing one prediction per row (alias
    ``prediction``), or ``None`` when the warehouse can't express the model:

    - linear models (``coef_`` + ``intercept_``) -> exact arithmetic SQL
      (``intercept + sum(coef_i * col_i)``), which matches local pandas scoring;
    - classification linear models -> ``None`` (the linear score isn't a class
      label — fall back to local);
    - tree / random-forest / k-means / isolation-forest -> ``None`` (documented
      PR-8 limit: clustering/scoring needs a different strategy).

    ``source`` is only required to exist; this function never pulls rows — it
    builds a string, so scoring happens entirely in the warehouse engine and no
    frame is materialized into the Python process.
    """
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    if coef is None or intercept is None:
        return None
    if task == "classification":
        return None
    coefs = [float(c) for c in coef]
    terms = [f"({float(intercept)})"]
    for col, c in zip(feature_cols, coefs, strict=False):
        terms.append(f"({c}) * CAST({sql_qcol('_f', col)} AS DOUBLE)")
    expr = " + ".join(terms)
    return f"SELECT {expr} AS prediction FROM ({frame_sql}) AS _f"
