"""Query planner: turns metric/dimension/filter requests into fan-out-safe SQL.

Single-table requests group by directly; multi-table requests use a
``JoinTree`` spanning tree. Star joins (one fact table + dimensions) join
directly; two-fact chains pre-aggregate each fact to the shared key's grain
via CTEs before joining (fan-out-safe).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

from demos.analytics.src.analytics.data_source import sql_qcol, sql_quote
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
    TimeColumn,
)

# Configured timezone for time-window filters (compare/trend). The data's time
# columns are assumed to be stored in this timezone; "now" is evaluated in it so
# period windows are stable regardless of server locale. Override via env.
_TIMEZONE = os.getenv("ANALYTICS_TIMEZONE", "UTC")
if not re.fullmatch(r"[A-Za-z0-9_+/.\-]+", _TIMEZONE or ""):
    _TIMEZONE = "UTC"


def _now_expr() -> str:
    """Current time evaluated in the configured timezone (wall clock)."""
    return f"current_timestamp AT TIME ZONE '{_TIMEZONE}'"


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
    # Derived metrics (ratios/shares/per-unit). Each is a dict with ``name`` and
    # ``expression``; the expression references metric refs (``table.col``) which
    # are rewritten to their aggregation (e.g. ``SUM(sales.amount)``) so ratios
    # stay additive-safe. e.g. {"name": "avg_price", "expression": "sales.amount/sales.quantity"}
    derivedMetrics: tuple[dict[str, str], ...] = ()


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

    select_metrics = _resolve_select_metrics(model, spec)

    if len(requested) == 1:
        sql = _group_by_query(next(iter(requested)), select_metrics, ds, where)
    else:
        tree = JoinTree.connect(model, requested, fact_tables)
        if len(tree.fact_tables) <= 1:
            sql = _star_join(tree, select_metrics, ds, where)
        else:
            if spec.derivedMetrics:
                raise ValueError(
                    "derived metrics (ratios) are only supported on a single table or star join; "
                    "use run_sql for cross-fact ratios"
                )
            sql = _fact_chain_join(tree, ms, ds, spec, model)

    aliases = [alias for _, alias in select_metrics]
    return _apply_order_limit(sql, aliases, ds, spec)


# ---------------------------------------------------------------------------
# Partial-failure robustness (PR-5)
# ---------------------------------------------------------------------------


class QueryPlanError(ValueError):
    """A scoped failure building a query plan, naming the offending table.

    Distinguishes a missing table from a schema-contract breach (a referenced
    column that disappeared) so callers / operators get an actionable message
    rather than a generic SQL error at execution time.
    """

    def __init__(self, table: str, reason: str) -> None:
        self.table = table
        self.reason = reason
        super().__init__(f"query plan failed for table '{table}': {reason}")


class MissingTableError(QueryPlanError):
    """A referenced table is absent from the live data source."""


class SchemaContractError(QueryPlanError):
    """A referenced column is absent from a table that otherwise exists."""


@dataclass
class PlanResult:
    """Result of planning a query, optionally with best-effort degradation."""

    sql: str
    warnings: list[str] = field(default_factory=list)
    dropped_tables: list[str] = field(default_factory=list)


def validate_plan(model: SemanticModel, spec: QuerySpec, source: Any) -> None:
    """Fail fast with a scoped ``QueryPlanError`` if ``spec`` can't be served.

    Checks, against the live ``source`` schema, that every referenced table
    exists and every referenced column (metric / dimension / filter / join key)
    is present. Raises :class:`MissingTableError` or :class:`SchemaContractError`
    naming the offending table and reason.
    """
    table_schemas = {t.name: t for t in source.tables()}
    ms = _resolve_metrics(model, spec.metrics)
    ds = _resolve_dimensions(model, spec.dimensions)
    requested = {m.table for m in ms} | {d.table for d in ds}

    def _cols(table: str) -> set[str]:
        return {c.name for c in table_schemas[table].columns}

    for m in ms:
        if m.table not in table_schemas:
            raise MissingTableError(
                m.table, f"metric '{m.ref}' references table '{m.table}' which is absent"
            )
        if m.column not in _cols(m.table):
            raise SchemaContractError(
                m.table, f"column '{m.column}' for metric '{m.ref}' is missing"
            )
    for d in ds:
        if d.table not in table_schemas:
            raise MissingTableError(
                d.table, f"dimension '{d.ref}' references table '{d.table}' which is absent"
            )
        if d.column not in _cols(d.table):
            raise SchemaContractError(
                d.table, f"column '{d.column}' for dimension '{d.ref}' is missing"
            )
    for f in spec.filters:
        ft = _filter_table(model, f.column)
        if ft is None or ft not in table_schemas:
            continue
        fc = f.column.split(".")[-1]
        if fc not in _cols(ft):
            raise SchemaContractError(ft, f"filter column '{fc}' is missing from table '{ft}'")

    # Join keys: only relevant when the plan spans more than one table.
    if len(requested) > 1:
        tree = JoinTree.connect(model, requested, {m.table for m in ms})
        for edge in tree.edges:
            rel = edge.relationship
            for tbl, cols in (
                (rel.from_table, rel.from_columns),
                (rel.to_table, rel.to_columns),
            ):
                if tbl not in table_schemas:
                    continue
                tcols = _cols(tbl)
                for c in cols:
                    if c not in tcols:
                        raise SchemaContractError(
                            tbl,
                            f"join column '{c}' on {rel.from_table}.{','.join(rel.from_columns)}"
                            f"~{rel.to_table}.{','.join(rel.to_columns)} is missing from '{tbl}'",
                        )


def _drop_table_refs(
    model: SemanticModel, spec: QuerySpec, table: str
) -> tuple[QuerySpec, int]:
    """Return a copy of ``spec`` with every reference to ``table`` removed.

    Drops metrics, dimensions, filters, and derived-metric expressions that name
    the table. Returns the spec and the number of references removed (0 means the
    table could not be isolated, so the caller should re-raise).
    """
    removed = 0
    new_metrics: list[str] = []
    for ref in spec.metrics:
        m = _resolve_metrics(model, (ref,))
        if m and m[0].table == table:
            removed += 1
            continue
        new_metrics.append(ref)
    new_dims: list[str] = []
    for ref in spec.dimensions:
        d = _resolve_dimensions(model, (ref,))
        if d and d[0].table == table:
            removed += 1
            continue
        new_dims.append(ref)
    new_filters = [
        f for f in spec.filters if _filter_table(model, f.column) != table
    ]
    removed += len(spec.filters) - len(new_filters)
    new_derived = [
        dm for dm in spec.derivedMetrics if table not in dm.get("expression", "")
    ]
    removed += len(spec.derivedMetrics) - len(new_derived)
    return (
        replace(
            spec,
            metrics=tuple(new_metrics),
            dimensions=tuple(new_dims),
            filters=tuple(new_filters),
            derivedMetrics=tuple(new_derived),
        ),
        removed,
    )


def plan_query(
    model: SemanticModel,
    spec: QuerySpec,
    source: Any = None,
    best_effort: bool = False,
) -> PlanResult:
    """Plan ``spec`` with partial-failure robustness.

    - With no ``source``: identical to :func:`plan` (pure SQL assembly).
    - With a ``source``: validates the plan against the live schema first, so a
      missing/broken table yields a scoped :class:`QueryPlanError` (naming the
      table) instead of a generic SQL failure.
    - With ``best_effort=True``: a failing table is dropped from the spec and the
      partial query is returned, with the dropped table listed in
      ``warnings`` / ``dropped_tables``.
    """
    if source is None:
        return PlanResult(sql=plan(model, spec))

    warnings: list[str] = []
    dropped: list[str] = []
    working = spec
    # Bounded loop: each iteration drops at most one table; stop once the plan
    # validates or nothing more can be dropped.
    for _ in range(len(model.metrics) + len(model.dimensions) + 2):
        try:
            validate_plan(model, working, source)
            break
        except QueryPlanError as exc:
            if not best_effort:
                raise
            working, removed = _drop_table_refs(model, working, exc.table)
            if removed == 0:
                raise
            warnings.append(f"dropped table '{exc.table}': {exc.reason}")
            dropped.append(exc.table)

    if not (working.metrics or working.dimensions):
        raise QueryPlanError(
            dropped[-1] if dropped else "?",
            "every requested table was dropped; nothing left to query",
        )
    return PlanResult(sql=plan(model, working), warnings=warnings, dropped_tables=dropped)


def feature_frame_sql(
    model: SemanticModel, target_table: str, columns: list[tuple[str, str]]
) -> str:
    """Row-grain feature frame: join ``target_table`` to related tables and select
    ``columns`` (``(table, column)`` pairs) at the target's row grain.

    Used by the modeling tools to assemble predictors from multiple tables via
    discovered relationships. Joins follow the same ``JoinTree`` as ``plan`` but
    without aggregation, so each target row keeps its own attributes (row-
    preserving for the usual many-to-one fact→dimension traversals).
    """
    tables = {target_table} | {t for (t, _c) in columns}
    tree = JoinTree.connect(model, tables, {target_table}, root=target_table)

    from_clause = sql_quote(tree.root)
    aliases: dict[str, str] = {}
    for edge in tree.edges:
        rel = edge.relationship
        # A many-to-many relationship would fan the target row out if joined
        # directly. Pre-aggregate the child to its join-key grain (one row per
        # key) before joining so the target keeps exactly one row per match.
        if rel.cardinality == "many_to_many":
            if edge.parent_is_from:
                key_cols = list(rel.from_columns)
                child = rel.to_table
                child_key = list(rel.to_columns)
            else:
                key_cols = list(rel.to_columns)
                child = rel.from_table
                child_key = list(rel.from_columns)
            child_col_list = [c for (tbl, c) in columns if tbl == child]
            group_cols = ", ".join(sql_qcol(child, ck) for ck in child_key)
            select_parts = [group_cols]
            for c in child_col_list:
                if c in child_key:
                    continue
                select_parts.append(f"MIN({sql_qcol(child, c)}) AS {sql_quote(c)}")
            child_sql = (
                f"SELECT {', '.join(select_parts)} FROM {sql_quote(child)} "
                f"GROUP BY {group_cols}"
            )
            alias = f"{child}__mm"
            on_parts = " AND ".join(
                f"{sql_qcol(rel.from_table, fc)} = {sql_quote(alias)}.{sql_quote(tc)}"
                if edge.parent_is_from
                else f"{sql_qcol(rel.to_table, tc)} = {sql_quote(alias)}.{sql_quote(fc)}"
                for fc, tc in zip(rel.from_columns, rel.to_columns, strict=False)
            )
            from_clause += f" JOIN ({child_sql}) AS {sql_quote(alias)} ON {on_parts}"
            aliases[child] = alias
        else:
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
            from_clause += f" JOIN {sql_quote(edge.child)} ON {on_parts}"

    select = ", ".join(sql_qcol(aliases.get(t, t), c) for (t, c) in columns)
    return f"SELECT {select} FROM {from_clause}".strip()


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
        root: str | None = None,
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

        def _bfs(root: str) -> tuple[set[str], dict[str, JoinEdge]]:
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
            return visited, parent_edge

        # Default: try each requested table as the BFS root and keep the one that
        # reaches the most requested tables (handles multi-hop paths through
        # intermediate tables that themselves may not be requested). When ``root``
        # is forced (e.g. a modeling target), anchor the tree there.
        best_root: str | None = None
        best_visited: set[str] = set()
        best_parent: dict[str, JoinEdge] = {}
        if root is not None:
            best_root, best_visited, best_parent = root, *_bfs(root)
        else:
            for cand in requested:
                visited, parent_edge = _bfs(cand)
                reached = len(requested & visited)
                if reached > len(requested & best_visited):
                    best_root, best_visited, best_parent = cand, visited, parent_edge

        missing = requested - best_visited
        if missing:
            available = ", ".join(
                f"{r.from_table}.{','.join(r.from_columns)}~{r.to_table}.{','.join(r.to_columns)}"
                for r in model.relationships
            )
            raise ValueError(
                f"no relationship path connects: {sorted(missing)}. "
                f"Discovered relationships: {available or 'none'}"
            )

        # Prune to requested tables
        tree_edges = _prune(best_parent, requested)
        plan_tables = {best_root}
        for e in tree_edges:
            plan_tables.add(e.child)

        if len(plan_tables) > MAX_JOIN_TABLES:
            raise ValueError(f"too many tables to join: {len(plan_tables)} (max {MAX_JOIN_TABLES})")

        return cls(
            root=best_root,
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


def _resolve_select_metrics(model: SemanticModel, spec: QuerySpec) -> list[tuple[str, str]]:
    """Resolve metrics (and any derived metrics) into ``(sql_expr, alias)`` pairs.

    Derived-metric expressions reference metric refs (``table.col`` or bare
    column); each is rewritten to its aggregation so ratios stay additive-safe,
    e.g. ``sales.amount/sales.quantity`` -> ``SUM("sales"."amount")/SUM("sales"."quantity")``.
    """
    out: list[tuple[str, str]] = []
    for m in _resolve_metrics(model, spec.metrics):
        out.append((f"{m.aggregation.upper()}({sql_qcol(m.table, m.column)})", m.column))
    for d in spec.derivedMetrics:
        name = d.get("name") or "derived"
        expr = d.get("expression", "")
        out.append((f"({_expand_expr(expr, model)})", name))
    return out


_METRIC_REF = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")


def _expand_expr(expr: str, model: SemanticModel) -> str:
    """Replace metric refs in ``expr`` with their aggregated SQL form."""
    agg_by_ref: dict[str, str] = {}
    for m in model.metrics:
        agg = f"{m.aggregation.upper()}({sql_qcol(m.table, m.column)})"
        agg_by_ref[m.ref.lower()] = agg
        agg_by_ref[m.column.lower()] = agg

    def _repl(match: re.Match[str]) -> str:
        return agg_by_ref.get(match.group(0).lower(), match.group(0))

    return _METRIC_REF.sub(_repl, expr)


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
    return _build_where_for_table(model, spec, None)


def _build_where_for_table(model: SemanticModel, spec: QuerySpec, table: str | None) -> str:
    """Build a WHERE clause.

    When ``table`` is ``None`` every filter/time clause is emitted (used by the
    single-table and star-join paths). When ``table`` names a table, only
    clauses that belong to that table are emitted (used by the multi-fact path,
    where each fact is pre-aggregated in its own CTE).
    """

    clauses: list[str] = []
    for f in spec.filters:
        if f.op not in OPERATORS:
            continue
        if table is not None and _filter_table(model, f.column) != table:
            continue
        clauses.append(f"{_safe_ref(f.column)} {f.op} '{_escape(f.value)}'")

    if spec.last_days is not None and spec.time_column:
        tc = _find_time_column(model, spec.time_column)
        if tc and (table is None or tc.table == table):
            ts_expr = tc.to_timestamp_sql(_safe_ref(spec.time_column))
            offset = spec.offset_days or 0
            clauses.append(
                f"{ts_expr} >= {_now_expr()} - INTERVAL '{spec.last_days + offset} days'"
            )
            if offset > 0:
                clauses.append(f"{ts_expr} < {_now_expr()} - INTERVAL '{offset} days'")

    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def _filter_table(model: SemanticModel, column: str) -> str | None:
    """Resolve which table a filter column belongs to (``table.col`` or bare)."""
    if "." in column:
        return column.split(".", 1)[0]
    for m in model.metrics:
        if m.column == column:
            return m.table
    for d in model.dimensions:
        if d.column == column:
            return d.table
    for ek in model.entity_keys:
        t, c = ek.split(".", 1)
        if c == column:
            return t
    return None


def _find_time_column(model: SemanticModel, ref: str) -> TimeColumn | None:
    for tc in model.time_columns:
        if tc.ref == ref or tc.column == ref:
            return tc
    return None


def _group_by_query(
    table: str,
    select_metrics: list[tuple[str, str]],
    dimensions: list[Dimension],
    where: str,
) -> str:
    select_parts = []
    for d in dimensions:
        if d.table == table:
            select_parts.append(sql_qcol(d.table, d.column))
    for expr, alias in select_metrics:
        select_parts.append(f"{expr} AS {sql_quote(alias)}")

    group_parts = [sql_qcol(d.table, d.column) for d in dimensions if d.table == table]

    select_clause = ", ".join(select_parts) if select_parts else "*"
    group_clause = f"GROUP BY {', '.join(group_parts)}" if group_parts else ""
    return f"SELECT {select_clause} FROM {sql_quote(table)} {where} {group_clause}".strip()


def _star_join(
    tree: JoinTree,
    select_metrics: list[tuple[str, str]],
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
    for expr, alias in select_metrics:
        select_parts.append(f"{expr} AS {sql_quote(alias)}")

    group_parts = [sql_qcol(d.table, d.column) for d in dimensions]

    select_clause = ", ".join(select_parts)
    group_clause = f"GROUP BY {', '.join(group_parts)}" if group_parts else ""
    return f"SELECT {select_clause} FROM {from_clause} {where} {group_clause}".strip()


def _fact_chain_join(
    tree: JoinTree,
    metrics: list[Metric],
    dimensions: list[Dimension],
    spec: QuerySpec,
    model: SemanticModel,
) -> str:
    """Pre-aggregate each table to its join-key grain via CTEs, then join.

    Every requested table (facts and any intermediate dimension tables) is
    pre-aggregated to the grain of its incident join keys plus the dimensions it
    owns. Because each CTE is already at the output grain, joining them on the
    relationship keys is fan-out-safe: a fact's pre-aggregated value is broadcast
    across the other table's finer dimensions rather than multiplied. The outer
    query just selects, without re-aggregating.
    """

    tables = sorted(tree.tables)

    # Map each table to the columns that link it to the rest of the tree.
    incident_keys: dict[str, list[str]] = {t: [] for t in tables}
    for edge in tree.edges:
        rel = edge.relationship
        if edge.parent_is_from:
            p_cols, c_cols = list(rel.from_columns), list(rel.to_columns)
        else:
            p_cols, c_cols = list(rel.to_columns), list(rel.from_columns)
        if edge.parent in incident_keys:
            incident_keys[edge.parent].extend(p_cols)
        if edge.child in incident_keys:
            incident_keys[edge.child].extend(c_cols)
    for t in incident_keys:
        seen: list[str] = []
        for c in incident_keys[t]:
            if c not in seen:
                seen.append(c)
        incident_keys[t] = seen

    ctes: list[str] = []
    for t in tables:
        t_metrics = [m for m in metrics if m.table == t]
        t_dims = [d.column for d in dimensions if d.table == t]
        group_cols = list(incident_keys[t]) + [c for c in t_dims if c not in incident_keys[t]]

        select_parts: list[str] = [sql_qcol(t, c) for c in group_cols]
        for m in t_metrics:
            select_parts.append(
                f"{m.aggregation.upper()}({sql_qcol(t, m.column)}) AS {sql_quote(m.column)}"
            )

        where_t = _build_where_for_table(model, spec, t)
        group_clause = (
            f"GROUP BY {', '.join(sql_qcol(t, c) for c in group_cols)}" if group_cols else ""
        )
        cte_sql = (
            f"{sql_quote(t)} AS ("
            f"SELECT {', '.join(select_parts)} FROM {sql_quote(t)} {where_t} {group_clause}"
            f")".strip()
        )
        ctes.append(cte_sql)

    from_clause = sql_quote(tree.root)
    for edge in tree.edges:
        rel = edge.relationship
        if edge.parent_is_from:
            on_parts = " AND ".join(
                f"{sql_qcol(edge.parent, fc)} = {sql_qcol(edge.child, tc)}"
                for fc, tc in zip(rel.from_columns, rel.to_columns, strict=False)
            )
        else:
            on_parts = " AND ".join(
                f"{sql_qcol(edge.child, tc)} = {sql_qcol(edge.parent, fc)}"
                for fc, tc in zip(rel.from_columns, rel.to_columns, strict=False)
            )
        from_clause += f" JOIN {sql_quote(edge.child)} ON {on_parts}"

    select_parts = [sql_qcol(d.table, d.column) for d in dimensions]
    for m in metrics:
        select_parts.append(sql_qcol(m.table, m.column))
    select_clause = ", ".join(select_parts) if select_parts else "*"

    sql = f"WITH {', '.join(ctes)} SELECT {select_clause} FROM {from_clause}".strip()
    return sql


def _apply_order_limit(
    sql: str, aliases: list[str], dimensions: list[Dimension], spec: QuerySpec
) -> str:
    result = sql
    # Order by: use the metric alias (column name), not the full table.column ref
    if spec.order_by:
        order_col = None
        if spec.order_by in aliases:
            order_col = sql_quote(spec.order_by)
        if order_col is None:
            for d in dimensions:
                if spec.order_by == d.ref or spec.order_by == d.column:
                    order_col = sql_qcol(d.table, d.column)
                    break
        if order_col is None:
            order_col = _safe_ref(spec.order_by)
        result += f" ORDER BY {order_col} {'DESC' if spec.descending else 'ASC'}"
    elif aliases:
        result += f" ORDER BY {sql_quote(aliases[0])} DESC"
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
