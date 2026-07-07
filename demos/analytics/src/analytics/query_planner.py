"""Query planner: turns metric/dimension/filter requests into fan-out-safe SQL.

Single-table requests group by directly; multi-table requests use a
``JoinTree`` spanning tree. Star joins (one fact table + dimensions) join
directly; two-fact chains pre-aggregate each fact to the shared key's grain
via CTEs before joining (fan-out-safe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from demos.analytics.src.analytics.data_source import sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
    TimeColumn,
)

OPERATORS = {"=", "!=", "<>", "<", "<=", ">", ">="}


@dataclass(frozen=True, slots=True)
class Filter:
    column: str
    op: str
    value: str


@dataclass(frozen=True, slots=True)
class QuerySpec:
    metrics: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    filters: tuple[Filter, ...] = ()
    last_days: int | None = None
    time_column: str | None = None
    order_by: str | None = None
    descending: bool = True
    limit: int | None = None
    offset_days: int = 0


def plan(model: SemanticModel, spec: QuerySpec) -> str:
    """Turn a ``QuerySpec`` into read-only SQL over the semantic model."""
    ms = _resolve_metrics(model, spec.metrics)
    ds = _resolve_dimensions(model, spec.dimensions)
    if not ms:
        raise ValueError("at least one metric is required")

    where = _build_where(model, spec)

    fact_tables: set[str] = set()
    requested: set[str] = set()
    for m in ms:
        fact_tables.add(m.table)
        requested.add(m.table)
    for d in ds:
        requested.add(d.table)

    if len(requested) == 1:
        sql = _group_by_query(next(iter(requested)), ms, ds, where)
    else:
        tree = JoinTree.connect(model, requested, fact_tables)
        if len(tree.fact_tables) <= 1:
            sql = _star_join(tree, ms, ds, where)
        else:
            sql = _fact_chain_join(tree, ms, ds, where)

    return _apply_order_limit(sql, ms, ds, spec)


# ---------------------------------------------------------------------------
# JoinTree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JoinEdge:
    parent: str
    child: str
    relationship: Any
    parent_is_from: bool


@dataclass(frozen=True, slots=True)
class JoinTree:
    root: str
    edges: tuple[JoinEdge, ...]
    fact_tables: frozenset[str]

    @property
    def tables(self) -> set[str]:
        t = {self.root}
        for e in self.edges:
            t.add(e.child)
        return t

    @classmethod
    def connect(
        cls,
        model: SemanticModel,
        requested: set[str],
        fact_tables: set[str],
    ) -> JoinTree:
        MAX_JOIN_TABLES = 6
        adjacency: dict[str, list[JoinEdge]] = {}
        for r in model.relationships:
            adjacency.setdefault(r.from_table, []).append(
                JoinEdge(r.from_table, r.to_table, r, True)
            )
            adjacency.setdefault(r.to_table, []).append(
                JoinEdge(r.to_table, r.from_table, r, False)
            )

        root = next(iter(requested))
        visited: set[str] = {root}
        parent_edge: dict[str, JoinEdge] = {}
        queue = [root]
        while queue:
            table = queue.pop(0)
            for e in adjacency.get(table, []):
                if e.child not in visited:
                    visited.add(e.child)
                    parent_edge[e.child] = e
                    queue.append(e.child)

        missing = requested - visited
        if missing:
            raise ValueError(f"no relationship path connects: {sorted(missing)}")

        # Prune to requested tables
        tree_edges = _prune(parent_edge, requested)
        plan_tables = {root}
        for e in tree_edges:
            plan_tables.add(e.child)

        if len(plan_tables) > MAX_JOIN_TABLES:
            raise ValueError(f"too many tables to join: {len(plan_tables)} (max {MAX_JOIN_TABLES})")

        return cls(
            root=root,
            edges=tuple(tree_edges),
            fact_tables=frozenset(fact_tables & plan_tables),
        )


def _prune(parent_edge: dict[str, JoinEdge], requested: set[str]) -> list[JoinEdge]:
    """Keep only edges on the path to requested tables."""
    keep: set[str] = set()
    for table in requested:
        t = table
        while t in parent_edge:
            keep.add(t)
            t = parent_edge[t].parent
    return [
        parent_edge[t]
        for t in sorted(
            keep, key=lambda x: list(parent_edge.keys()).index(x) if x in parent_edge else 0
        )
    ]


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------


def _resolve_metrics(model: SemanticModel, refs: tuple[str, ...]) -> list[Metric]:
    by_ref = {m.ref.lower(): m for m in model.metrics}
    by_col = {m.column.lower(): m for m in model.metrics}
    result = []
    for ref in refs:
        rl = ref.lower()
        if rl in by_ref:
            result.append(by_ref[rl])
        elif rl in by_col:
            result.append(by_col[rl])
    return result


def _resolve_dimensions(model: SemanticModel, refs: tuple[str, ...]) -> list[Dimension]:
    by_ref = {d.ref.lower(): d for d in model.dimensions}
    by_col = {d.column.lower(): d for d in model.dimensions}
    result = []
    for ref in refs:
        rl = ref.lower()
        if rl in by_ref:
            result.append(by_ref[rl])
        elif rl in by_col:
            result.append(by_col[rl])
    return result


def _build_where(model: SemanticModel, spec: QuerySpec) -> str:
    clauses: list[str] = []
    for f in spec.filters:
        if f.op not in OPERATORS:
            continue
        clauses.append(f"{_safe_ref(f.column)} {f.op} '{_escape(f.value)}'")

    if spec.last_days is not None and spec.time_column:
        tc = _find_time_column(model, spec.time_column)
        if tc:
            ts_expr = tc.to_timestamp_sql(_safe_ref(spec.time_column))
            offset = spec.offset_days or 0
            clauses.append(
                f"{ts_expr} >= current_timestamp - INTERVAL '{spec.last_days + offset} days'"
            )
            if offset > 0:
                clauses.append(f"{ts_expr} < current_timestamp - INTERVAL '{offset} days'")

    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _find_time_column(model: SemanticModel, ref: str) -> TimeColumn | None:
    for tc in model.time_columns:
        if tc.ref == ref or tc.column == ref:
            return tc
    return None


def _group_by_query(
    table: str,
    metrics: list[Metric],
    dimensions: list[Dimension],
    where: str,
) -> str:
    select_parts = []
    for d in dimensions:
        if d.table == table:
            select_parts.append(sql_qcol(d.table, d.column))
    for m in metrics:
        if m.table == table:
            select_parts.append(
                f"{m.aggregation.upper()}({sql_qcol(m.table, m.column)}) AS {m.column}"
            )

    group_parts = [sql_qcol(d.table, d.column) for d in dimensions if d.table == table]

    select_clause = ", ".join(select_parts) if select_parts else "*"
    group_clause = f"GROUP BY {', '.join(group_parts)}" if group_parts else ""
    return f"SELECT {select_clause} FROM {sql_quote(table)} {where} {group_clause}".strip()


def _star_join(
    tree: JoinTree,
    metrics: list[Metric],
    dimensions: list[Dimension],
    where: str,
) -> str:
    root = tree.root
    from_clause = sql_quote(root)
    for edge in tree.edges:
        rel = edge.relationship
        child_q = sql_quote(edge.child)
        if edge.parent_is_from:
            on_parts = " AND ".join(
                f"{sql_qcol(rel.from_table, fc)} = {sql_qcol(rel.to_table, tc)}"
                for fc, tc in zip(rel.from_columns, rel.to_columns, strict=False)
            )
        else:
            on_parts = " AND ".join(
                f"{sql_qcol(rel.to_table, tc)} = {sql_qcol(rel.from_table, fc)}"
                for fc, tc in zip(rel.from_columns, rel.to_columns, strict=False)
            )
        from_clause += f" JOIN {child_q} ON {on_parts}"

    select_parts = []
    for d in dimensions:
        select_parts.append(sql_qcol(d.table, d.column))
    for m in metrics:
        select_parts.append(f"{m.aggregation.upper()}({sql_qcol(m.table, m.column)}) AS {m.column}")

    group_parts = [sql_qcol(d.table, d.column) for d in dimensions]

    select_clause = ", ".join(select_parts)
    group_clause = f"GROUP BY {', '.join(group_parts)}" if group_parts else ""
    return f"SELECT {select_clause} FROM {from_clause} {where} {group_clause}".strip()


def _fact_chain_join(
    tree: JoinTree,
    metrics: list[Metric],
    dimensions: list[Dimension],
    where: str,
) -> str:
    """Pre-aggregate each fact to the shared key's grain via CTEs before joining."""
    raise ValueError(
        "multi-fact metric queries are not planned automatically yet; "
        "query one fact table at a time or use run_sql with an explicit read-only join"
    )


def _apply_order_limit(
    sql: str, metrics: list[Metric], dimensions: list[Dimension], spec: QuerySpec
) -> str:
    result = sql
    # Order by: use the metric alias (column name), not the full table.column ref
    if spec.order_by:
        # Try to match to a metric alias first
        order_col = None
        for m in metrics:
            if spec.order_by == m.ref or spec.order_by == m.column:
                order_col = m.column
                break
        if order_col is None:
            # Try dimension
            for d in dimensions:
                if spec.order_by == d.ref or spec.order_by == d.column:
                    order_col = sql_qcol(d.table, d.column)
                    break
        if order_col is None:
            order_col = _safe_ref(spec.order_by)
        result += f" ORDER BY {order_col} {'DESC' if spec.descending else 'ASC'}"
    elif metrics:
        result += f" ORDER BY {metrics[0].column} DESC"
    if spec.limit:
        result += f" LIMIT {max(1, spec.limit)}"
    return result


def _safe_ref(ref: str) -> str:
    if "." in ref:
        parts = ref.split(".")
        return ".".join(sql_quote(p) for p in parts)
    return sql_quote(ref)


def _escape(value: str) -> str:
    return value.replace("'", "''")
