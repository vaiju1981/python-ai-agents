"""DSL query IR (PR-D2).

``DslQuery`` / ``DslFilter`` are the parsed intermediate representation of the
textual analytics DSL. They are deliberately 1:1 with ``query_planner.QuerySpec``
plus inline derived metrics, so ``to_spec`` can hand them straight to the
existing planner. Name resolution (catalog metric names, dimensions, time
column) happens in ``to_spec`` using the ``SemanticModel`` and ``MetricCatalog``
built in PR-D1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from demos.analytics.src.analytics.dsl.catalog import (
    BaseMetricDef,
    MetricCatalog,
)
from demos.analytics.src.analytics.dsl.grounding import NameResolver
from demos.analytics.src.analytics.semantic_model import Metric, SemanticModel


@dataclass(frozen=True, slots=True)
class DslFilter:
    column: str
    op: str
    value: str | list[str]


@dataclass(frozen=True, slots=True)
class DslQuery:
    metrics: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    filters: tuple[DslFilter, ...] = ()
    last_days: int | None = None
    time_column: str | None = None
    between_start: str | None = None
    between_end: str | None = None
    order_by: str | None = None
    descending: bool = True
    limit: int | None = None

    def to_spec(
        self,
        model: SemanticModel,
        catalog: MetricCatalog | None = None,
        resolver: NameResolver | None = None,
    ):
        """Compile this DSL query into a ``query_planner.QuerySpec``.

        Metric tokens are resolved through the catalog (PR-D1): a base metric
        becomes a plain ``table.column`` ref; a calculated metric (or an inline
        ``expr AS alias``) becomes a ``derivedMetrics`` entry whose expression is
        already expanded to additive-safe aggregated SQL. Dimensions and the time
        column are resolved against the ``SemanticModel``. When a ``resolver``
        (PR-D3) is supplied, friendly names are grounded *before* catalog/model
        resolution, and filter values are normalized per-dimension.
        """
        from demos.analytics.src.analytics.query_planner import Filter, QuerySpec

        # Map each metric token to its output SELECT alias so ORDER BY (and the
        # engine's result renaming) can use the (possibly source-derived) alias,
        # e.g. a synonym ``revenue`` grounded to ``sales.amount`` selects as
        # ``amount``.
        metric_aliases = metric_token_aliases(self.metrics, model, catalog, resolver)
        metrics: list[str] = []
        derived: list[dict[str, str]] = []
        for tok in self.metrics:
            name, expr = _split_alias(tok)
            kind, a, b = _resolve_metric_token(tok, model, catalog, resolver)
            if kind == "metric":
                metrics.append(a)
            else:
                derived.append({"name": a, "expression": b})

        order_by = _resolve_order_by(self.order_by, metric_aliases, model, resolver)

        dimensions: list[str] = []
        for d in self.dimensions:
            dimensions.append(_resolve_dimension(d, model, resolver))

        filters = tuple(
            Filter(
                column=_resolve_column(f.column, model, resolver),
                op=f.op,
                value=_normalize_value(f.column, f.value, model, resolver),
            )
            for f in self.filters
        )

        time_column = self.time_column
        if time_column is None and (self.last_days is not None or self.between_start is not None):
            if model.time_columns:
                time_column = model.time_columns[0].ref

        return QuerySpec(
            metrics=tuple(metrics),
            dimensions=tuple(dimensions),
            filters=filters,
            last_days=self.last_days,
            time_column=time_column,
            between_start=self.between_start,
            between_end=self.between_end,
            order_by=order_by,
            descending=self.descending,
            limit=self.limit,
            derivedMetrics=tuple(derived),
        )

    def to_text(self) -> str:
        """Render a canonical, normalized form of the query (round-trip)."""
        parts: list[str] = ["SELECT " + ", ".join(_quote_ident(m) for m in self.metrics)]
        if self.dimensions:
            parts.append("BY " + ", ".join(_quote_ident(d) for d in self.dimensions))
        if self.filters:
            parts.append("WHERE " + " AND ".join(_render_filter(f) for f in self.filters))
        if self.last_days is not None:
            parts.append(f"SINCE {self.last_days} DAYS")
        if self.between_start is not None:
            end = self.between_end or self.between_start
            parts.append(f"BETWEEN {self.between_start} AND {end}")
        if self.order_by is not None:
            dirn = "DESC" if self.descending else "ASC"
            parts.append(f"ORDER BY {self.order_by} {dirn}")
        if self.limit is not None:
            parts.append(f"LIMIT {self.limit}")
        return " ".join(parts)


def _render_value(value: str) -> str:
    """Quote a filter literal if it isn't a bare identifier or number."""
    if value.isdigit() or (value and (value[0].isalpha() or value[0] == "_")
                           and all(c.isalnum() or c == "_" for c in value)):
        return value
    return "'" + value.replace("'", "\\'") + "'"


def _quote_ident(s: str) -> str:
    """Quote an identifier (metric/dimension/alias) for round-trip if it is not a
    single bare token (e.g. a multi-word business name like ``net win``)."""
    if s and (s[0].isalpha() or s[0] == "_") and all(c.isalnum() or c == "_" for c in s):
        return s
    return '"' + s.replace('"', '\\"') + '"'


def _render_filter(f: DslFilter) -> str:
    value = f.value
    if isinstance(value, (list, tuple)):
        return f"{_quote_ident(f.column)} IN (" + ", ".join(_render_value(v) for v in value) + ")"
    return f"{_quote_ident(f.column)} {f.op} {_render_value(value)}"


def metric_token_aliases(
    tokens: tuple[str, ...],
    model: SemanticModel,
    catalog: MetricCatalog | None,
    resolver: NameResolver | None = None,
) -> dict[str, str]:
    """Map each DSL metric token to its output SELECT alias.

    A base metric grounded to ``table.column`` selects under its source column
    name; a calculated/inline metric selects under its declared name.
    """
    out: dict[str, str] = {}
    for tok in tokens:
        name, expr = _split_alias(tok)
        if expr is not None:
            out[tok] = name  # inline ``expr AS alias``
            continue
        kind, a, _ = _resolve_metric_token(tok, model, catalog, resolver)
        if kind == "metric":
            out[tok] = _column_of_ref(a, model)
        else:
            out[tok] = name
    return out


def _column_of_ref(ref: str, model: SemanticModel) -> str:
    """Output SELECT alias for a base ``table.column`` ref (its column name)."""
    for m in model.metrics:
        if m.ref.lower() == ref.lower():
            return m.column
    return ref.split(".")[-1]


def _resolve_order_by(
    order_by: str | None,
    metric_aliases: dict[str, str],
    model: SemanticModel,
    resolver: NameResolver | None,
) -> str | None:
    """Rewrite ORDER BY to the resolved metric alias or dimension ref.

    A friendly/synonym metric (e.g. ``revenue``) is rewritten to its source
    column alias (``amount``); a ``table.column`` ref token is left as-is; a
    dimension token is grounded to its ref.
    """
    if order_by is None:
        return None
    if order_by in metric_aliases:
        alias = metric_aliases[order_by]
        if "." not in order_by:
            return alias
        return order_by
    # Otherwise treat it as a dimension ref (grounded if a resolver is present).
    return _resolve_dimension(order_by, model, resolver)


def _split_alias(token: str) -> tuple[str, str | None]:
    """Split ``expr AS alias`` -> (alias, expr); plain name -> (name, None)."""
    parts = re.split(r"\s+AS\s+", token, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[1].strip(), parts[0].strip()
    return token.strip(), None


def _base_from_model(model: SemanticModel, name: str) -> Metric | None:
    nl = name.lower()
    for m in model.metrics:
        if m.ref.lower() == nl or m.column.lower() == nl:
            return m
    return None


def _resolve_metric_token(
    token: str,
    model: SemanticModel,
    catalog: MetricCatalog | None,
    resolver: NameResolver | None = None,
) -> tuple[str, str, str]:
    """Return (kind, a, b): kind 'metric' -> (_, ref, _); 'derived' -> (_, name, sql)."""
    name, expr = _split_alias(token)
    if expr is None:
        if resolver is not None:
            defn = resolver.resolve_metric(name, catalog, model)
        elif catalog is not None:
            defn = catalog.get(name)
        else:
            defn = None
        if defn is not None:
            if isinstance(defn, BaseMetricDef):
                return ("metric", f"{defn.table}.{defn.column}", "")
            sql, _ = catalog.resolve(defn.name, model)
            return ("derived", defn.name, sql)
        base = _base_from_model(model, name)
        if base is not None:
            return ("metric", base.ref, "")
        from demos.analytics.src.analytics.dsl.parser import DslParseError

        raise DslParseError(f"unknown metric '{name}'")
    # Inline expr AS alias: expand catalog names to aggregated SQL.
    if catalog is not None:
        expression = catalog.expand(expr, model)
    else:
        expression = expr
    return ("derived", name, expression)


def _resolve_dimension(
    token: str, model: SemanticModel, resolver: NameResolver | None = None
) -> str:
    if resolver is not None:
        return resolver.resolve_dimension(token, model).ref
    nl = token.lower()
    for d in model.dimensions:
        if d.ref.lower() == nl or d.column.lower() == nl:
            return d.ref
    from demos.analytics.src.analytics.dsl.parser import DslParseError

    raise DslParseError(f"unknown dimension '{token}'")


def _resolve_column(
    token: str, model: SemanticModel, resolver: NameResolver | None = None
) -> str:
    """Resolve a filter/operand column to its qualified ``table.column`` ref.

    Falls back to the bare token when it matches no known dimension or metric
    (e.g. a metric-ref filter like ``sales.amount`` that is not a dimension).
    """
    if resolver is not None:
        try:
            return resolver.resolve_dimension(token, model).ref
        except Exception:
            pass
    nl = token.lower()
    for d in model.dimensions:
        if d.ref.lower() == nl or d.column.lower() == nl:
            return d.ref
    for m in model.metrics:
        if m.ref.lower() == nl or m.column.lower() == nl:
            return m.ref
    return token


def _normalize_value(
    column_token: str,
    value: str | list[str],
    model: SemanticModel,
    resolver: NameResolver | None,
) -> str | list[str]:
    """Normalize a filter literal via the per-dimension value map, if any.

    Only applies when ``column_token`` resolves to a known dimension; metric-ref
    filters are passed through untouched.
    """
    if resolver is None:
        return value
    try:
        dim_ref = resolver.resolve_dimension(column_token, model).ref
    except Exception:
        dim_ref = _resolve_dimension(column_token, model)
    if isinstance(value, (list, tuple)):
        return [resolver.resolve_value(dim_ref, v) for v in value]
    return resolver.resolve_value(dim_ref, value)
