"""Column and dataset profiling over any ``DataSource``.

Uses DuckDB SQL (via the source's ``native_query``) for all statistics — fast
and works with any backend that supports standard SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.data_source import (
    ColumnRole,
    ColumnSchema,
    DataSource,
    Relationship,
    TableSchema,
    sql_quote,
)
from demos.analytics.src.analytics.relationships import discover as discover_relationships


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    table: str
    name: str
    physical_type: str
    rows: int
    distinct: int
    nulls: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    stddev: float | None = None
    sample_values: tuple[str, ...] = ()
    signals: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    tables: tuple[TableSchema, ...]
    columns: tuple[ColumnProfile, ...]
    relationships: tuple[Relationship, ...]
    import_plan: dict[str, dict[str, str]] = field(default_factory=dict)


_DATE_RE = re.compile(r"\d{4}[/-]\d{2}[/-]\d{2}.*")
_LEADING_ZERO_RE = re.compile(r"0\d+")
_BOOLS = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"})


def profile_column(source: DataSource, table: str, col: ColumnSchema) -> ColumnProfile:
    """Profile a single column using DuckDB SQL."""
    c = sql_quote(col.name)
    t = sql_quote(table)
    numeric = is_numeric(col.physical_type)

    base = (
        f"SELECT COUNT(*) AS n_rows, COUNT(DISTINCT {c}) AS n_distinct, "
        f"COUNT(*) FILTER (WHERE {c} IS NULL) AS n_nulls"
    )
    if numeric:
        cd = f"CAST({c} AS DOUBLE)"
        mma = f", MIN({cd}) AS mn, MAX({cd}) AS mx, AVG({cd}) AS av"
        sd = f", stddev_pop({cd}) AS sd"
        agg = _first_ok(
            source,
            [
                f"{base}{mma}{sd} FROM {t}",
                f"{base}{mma} FROM {t}",
                f"{base} FROM {t}",
            ],
        )
    else:
        agg = source.native_query(f"{base} FROM {t}")[0]

    # Sample values
    sample_rows = source.native_query(
        f"SELECT DISTINCT {c} AS v FROM {t} WHERE {c} IS NOT NULL LIMIT 20"
    )
    samples = tuple(str(r["v"]) for r in sample_rows)

    rows = int(agg.get("n_rows", 0))
    distinct = int(agg.get("n_distinct", 0))
    nulls = int(agg.get("n_nulls", 0))

    # Signals
    signals = set()
    if any(_DATE_RE.match(s) for s in samples):
        signals.add("date-like")
    if any(_LEADING_ZERO_RE.match(s) for s in samples):
        signals.add("leading-zeros")
    if distinct == 2 and all(s.lower() in _BOOLS for s in samples):
        signals.add("bool-like")

    name_lower = col.name.lower()
    name_id = bool(re.match(r".*(id|key|code|uuid|guid)$", name_lower))
    high_card = not is_floating_point(col.physical_type) and rows >= 50 and distinct / rows > 0.95
    if name_id or high_card:
        signals.add("id-like")

    min_val = _to_float(agg.get("mn")) if numeric else None
    max_val = _to_float(agg.get("mx")) if numeric else None
    name_time_hint = bool(
        re.search(r"(time|date|epoch|ts|created|updated|timestamp|at)$", name_lower)
    )
    if (
        numeric
        and "id-like" not in signals
        and min_val is not None
        and min_val >= 1e9
        and max_val is not None
        and max_val <= 5e12
        and (distinct > 10 or name_time_hint)
    ):
        signals.add("epoch-like")

    return ColumnProfile(
        table=table,
        name=col.name,
        physical_type=col.physical_type,
        rows=rows,
        distinct=distinct,
        nulls=nulls,
        min=min_val,
        max=max_val,
        mean=_to_float(agg.get("av")) if numeric else None,
        stddev=_to_float(agg.get("sd")) if numeric else None,
        sample_values=samples,
        signals=frozenset(signals),
    )


def profile_dataset(
    source: DataSource,
    catalog: Any | None = None,
) -> DatasetProfile:
    """Profile all tables and discover relationships."""
    tables = source.tables()
    columns: list[ColumnProfile] = []
    roles_by_table: dict[str, dict[str, ColumnRole]] = {}
    import_plan: dict[str, dict[str, str]] = {}

    typed_tables: list[TableSchema] = []
    for table in tables:
        typed_cols: list[ColumnSchema] = []
        role_map: dict[str, ColumnRole] = {}
        for col in table.columns:
            cp = profile_column(source, table.name, col)
            columns.append(cp)
            from demos.analytics.src.analytics.semantic_roles import classify_role

            role = classify_role(cp)
            typed_cols.append(
                ColumnSchema(name=col.name, physical_type=col.physical_type, role=role)
            )
            role_map[col.name] = role
            if "leading-zeros" in cp.signals:
                import_plan.setdefault(table.name, {})[col.name] = "VARCHAR"
        roles_by_table[table.name] = role_map
        typed_tables.append(
            TableSchema(name=table.name, rows=table.rows, columns=tuple(typed_cols))
        )

    stats_by_ref = {f"{cp.table}.{cp.name}": cp for cp in columns}
    relationships = discover_relationships(source, roles_by_table, stats_by_ref)
    if catalog is not None:
        relationships = catalog.apply(relationships)

    return DatasetProfile(
        tables=tuple(typed_tables),
        columns=tuple(columns),
        relationships=tuple(relationships),
        import_plan=import_plan,
    )


def is_numeric(duck_type: str) -> bool:
    s = duck_type.upper()
    return any(k in s for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC", "HUGEINT"))


def is_floating_point(duck_type: str) -> bool:
    s = duck_type.upper()
    return any(k in s for k in ("DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC"))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_ok(source: DataSource, sqls: list[str]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for sql in sqls:
        try:
            return source.native_query(sql)[0]
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return {}
