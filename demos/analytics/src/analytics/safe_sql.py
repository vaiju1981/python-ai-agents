"""Safe SQL validation for the ``run_sql`` tool.

Replaces the keyword-scanner heuristic with a real parser when ``sqlglot`` is
available: we parse the AST, reject multiple statements, and reject any
statement that is not a read-only query (SELECT / WITH / SHOW / DESCRIBE /
EXPLAIN), plus file-reading functions. If ``sqlglot`` is absent or cannot parse
a particular dialect-specific query, we fall back to the keyword scanner so we
never block a legitimate read.

The governance boundary (no writes) is the point: the tool is READ_ONLY and the
underlying DuckDB connection also disables external file access after import.
"""

from __future__ import annotations

import re

_ALLOWED_TYPES = ("Select", "Show", "Describe", "Explain", "Values")
# Statement/expression types that are never allowed through run_sql.
_FORBIDDEN_TYPES = (
    "Insert",
    "Update",
    "Delete",
    "Create",
    "Drop",
    "Alter",
    "Set",
    "Grant",
    "Revoke",
    "Copy",
    "Merge",
    "Truncate",
    "Command",
    "ReadCSV",
    "ReadParquet",
)
# Bare function names that read from disk / external systems.
_FORBIDDEN_FUNCS = {"read_csv", "read_csv_auto", "read_parquet", "read_json", "read_blob"}
# Parser-independent guard: file-reading table functions written as
# ``FROM read_csv('/etc/passwd')`` parse as a Table (not a Func), so the AST
# check above alone can miss them when sqlglot is absent or fails to parse.
# This regex catches the call form regardless of the parser.
_FILE_READ_RE = re.compile(
    r"\b(?:read_csv|read_csv_auto|read_parquet|read_parquet_auto|"
    r"read_json|read_json_auto|read_blob)\s*\(",
    re.IGNORECASE,
)


def safe_sql_error(sql: str) -> str | None:
    """Return an error message if ``sql`` is not a safe read-only query, else None."""
    # Fast, parser-independent block on file-reading table functions. This works
    # even when sqlglot is not installed (the AST path below would be skipped).
    if _FILE_READ_RE.search(sql):
        return "forbidden SQL function: file read (read_csv/read_parquet/read_json/...)"

    try:
        import sqlglot
    except Exception:
        return _fallback(sql)

    try:
        parsed = sqlglot.parse(sql, read="duckdb")
    except Exception:
        # Could not parse (dialect-specific syntax) — don't block; use heuristic.
        return _fallback(sql)

    statements = [s for s in parsed if s is not None]
    if len(statements) > 1:
        return "multiple SQL statements are not allowed"
    if not statements:
        return "sql must contain a read-only statement"

    stmt = statements[0]
    if type(stmt).__name__ not in _ALLOWED_TYPES:
        return f"forbidden SQL statement: {type(stmt).__name__}"

    # Reject any forbidden node anywhere in the tree (e.g. a CTE writing, or a
    # file-reading function call nested in a SELECT).
    for node in stmt.walk():
        name = type(node).__name__
        if name in _FORBIDDEN_TYPES:
            return f"forbidden SQL construct: {name}"
        if name == "Func" and getattr(node, "name", "").lower() in _FORBIDDEN_FUNCS:
            return f"forbidden SQL function: {getattr(node, 'name', '')}"

    return None


def _fallback(sql: str) -> str | None:
    # Imported lazily to avoid a circular import with toolset.
    from demos.analytics.src.analytics.toolset import _read_only_sql_error

    return _read_only_sql_error(sql)
