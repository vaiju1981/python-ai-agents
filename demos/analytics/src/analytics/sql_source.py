"""SQL data source: DuckDB files, SQLite, Postgres, or any SQLAlchemy URL.

Tables are read via DuckDB's ``ATTACH`` for supported engines, or directly for
DuckDB native files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import duckdb

from demos.analytics.src.analytics.data_source import (
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
    sql_qtable,
)

# Aliases for ATTACH become raw SQL identifiers; restrict to a safe shape.
_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SqlSource(DataSource):
    """DuckDB-backed SQL data source for DuckDB files, SQLite, or attached databases."""

    def __init__(self, db_path: str = ":memory:", attach: dict[str, str] | None = None) -> None:
        """Initialize with a DuckDB file, optionally attaching external databases.

        Args:
            db_path: Path to a DuckDB file (``:memory:`` for ephemeral).
            attach: Optional dict of ``{alias: connection_string}`` to ATTACH
                external databases (e.g. ``{"pg": "postgresql://..."}``).
        """
        self._conn = duckdb.connect(db_path)
        self._table_names: list[str] = []
        self._attached_prefix: dict[str, str] = {}

        if attach:
            for alias, conn_str in attach.items():
                if not _ALIAS_RE.match(alias):
                    raise ValueError(f"attach alias '{alias}' is not a safe SQL identifier")
                self._conn.execute(f"ATTACH '{conn_str}' AS {alias} (READ_ONLY)")
                self._attached_prefix[alias] = f"{alias}."

        self._discover_tables()

    def _discover_tables(self) -> None:
        # ``duckdb_tables()`` reports every table with its owning catalog, so we
        # can unambiguously separate host tables from attached-warehouse tables
        # (attached tables otherwise surface under the host's ``main`` schema in
        # ``information_schema``). Attached tables are referenced as
        # ``catalog.schema.table`` (or ``catalog.table`` when the schema is main).
        try:
            main_db = self._conn.execute("SELECT current_database()").fetchone()[0]
        except Exception:
            main_db = "memory"
        try:
            rows = self._conn.execute(
                "SELECT database_name, schema_name, table_name FROM duckdb_tables() "
                "WHERE schema_name NOT IN ('information_schema', 'pg_catalog')"
            ).fetchall()
        except Exception:
            rows = []
        for database, schema, name in rows:
            if name.startswith("duckdb_") or name.startswith("sqlite_"):
                continue
            if database == main_db:
                full = name if schema == "main" else f"{schema}.{name}"
            elif schema == "main":
                full = f"{database}.{name}"
            else:
                full = f"{database}.{schema}.{name}"
            self._table_names.append(full)

    def tables(self) -> list[TableSchema]:
        result = []
        for name in self._table_names:
            cols = self._table_columns(name)
            row_count = self._conn.execute(f"SELECT COUNT(*) FROM {sql_qtable(name)}").fetchone()[0]
            result.append(TableSchema(name=name, rows=row_count, columns=tuple(cols)))
        return result

    def _table_columns(self, table: str) -> list[ColumnSchema]:
        schema = self._conn.execute(f"DESCRIBE {sql_qtable(table)}").fetchall()
        return [ColumnSchema(name=row[0], physical_type=row[1]) for row in schema]

    def relationships(self) -> list[Relationship]:
        return []

    def sample(self, table: str, limit: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"SELECT * FROM {sql_qtable(table)} LIMIT {max(1, limit)}"
        ).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def native_query(self, sql: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql).fetchall()
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    def native_query_with_limit(self, sql: str, max_rows: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(sql).fetchmany(max_rows)
        cols = [d[0] for d in self._conn.description]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    # --- ingestion seam (PR-12) ---------------------------------------------
    # SqlSource is a read-only backend (attached warehouses are READ_ONLY). Use
    # CsvSource for incremental ingestion. ``row_count`` is provided because it
    # is a pure read.
    def row_count(self, table: str) -> int:
        return self._conn.execute(
            f"SELECT COUNT(*) FROM {sql_qtable(table)}"
        ).fetchone()[0]

    def append_rows(self, table: str, rows: list[dict[str, Any]]) -> int:
        raise NotImplementedError("SqlSource is read-only; use CsvSource for ingestion (PR-12)")

    def ingest_csv(self, table: str, csv_path: Path, *, mode: str = "append") -> int:
        raise NotImplementedError("SqlSource is read-only; use CsvSource for ingestion (PR-12)")

    def upsert(self, table: str, rows: list[dict[str, Any]], keys: list[str]) -> int:
        raise NotImplementedError("SqlSource is read-only; use CsvSource for ingestion (PR-12)")

    def close(self) -> None:
        self._conn.close()
