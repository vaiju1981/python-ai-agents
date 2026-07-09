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
        return SqlSource(db_path=db_path, attach={alias: conn_str})
    except Exception as exc:  # pragma: no cover - depends on installed DuckDB extensions
        raise RuntimeError(
            f"could not attach {kind} warehouse (is the DuckDB "
            f"{kind} extension installed?): {redact_secrets(str(exc))}"
        ) from exc


def available_kinds() -> list[str]:
    return sorted(_ATTACH_TEMPLATES)
