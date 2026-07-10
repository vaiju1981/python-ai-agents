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
from dataclasses import dataclass, field

from demos.analytics.src.analytics.dsl.catalog import (
    BaseMetricDef,
    CalculatedMetricDef,
    MetricCatalog,
)
from demos.analytics.src.analytics.semantic_model import Dimension, Metric, SemanticModel


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

    def to_spec(self, model: SemanticModel, catalog: MetricCatalog | None = None):
        """Compile this DSL query into a ``query_planner.QuerySpec``.

        Metric tokens are resolved through the catalog (PR-D1): a base metric
        becomes a plain ``table.column`` ref; a calculated metric (or an inline
        ``expr AS alias``) becomes a ``derivedMetrics`` entry whose expression is
        already expanded to additive-safe aggregated SQL. Dimensions and the time
        column are resolved against the ``SemanticModel``.
        """
        from demos.analytics.src.analytics.query_planner import Filter, QuerySpec

        metrics: list[str] = []
        derived: list[dict[str, str]] = []
        for tok in self.metrics:
            kind, a, b = _resolve_metric_token(tok, model, catalog)
            if kind == "metric":
                metrics.append(a)
            else:
                derived.append({"name": a, "expression": b})

        dimensions: list[str] = []
        for d in self.dimensions:
            dimensions.append(_resolve_dimension(d, model))

        filters = tuple(Filter(column=f.column, op=f.op, value=f.value) for f in self.filters)

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
            order_by=self.order_by,
            descending=self.descending,
            limit=self.limit,
            derivedMetrics=tuple(derived),
        )

    def to_text(self) -> str:
        """Render a canonical, normalized form of the query (round-trip)."""
        parts: list[str] = ["SELECT " + ", ".join(self.metrics)]
        if self.dimensions:
            parts.append("BY " + ", ".join(self.dimensions))
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


def _render_filter(f: "DslFilter") -> str:
    value = f.value
    if isinstance(value, (list, tuple)):
        return f"{f.column} IN (" + ", ".join(_render_value(v) for v in value) + ")"
    return f"{f.column} {f.op} {_render_value(value)}"


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
    token: str, model: SemanticModel, catalog: MetricCatalog | None
) -> tuple[str, str, str]:
    """Return (kind, a, b): kind 'metric' -> (_, ref, _); 'derived' -> (_, name, sql)."""
    name, expr = _split_alias(token)
    if expr is None:
        if catalog is not None:
            defn = catalog.get(name)
            if defn is not None:
                if isinstance(defn, BaseMetricDef):
                    return ("metric", defn.ref, "")
                sql, _ = catalog.resolve(name, model)
                return ("derived", name, sql)
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


def _resolve_dimension(token: str, model: SemanticModel) -> str:
    nl = token.lower()
    for d in model.dimensions:
        if d.ref.lower() == nl or d.column.lower() == nl:
            return d.ref
    from demos.analytics.src.analytics.dsl.parser import DslParseError

    raise DslParseError(f"unknown dimension '{token}'")
