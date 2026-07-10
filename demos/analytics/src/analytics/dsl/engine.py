"""DSL engine orchestrator + planner integration (PR-D4).

This is the single entry point that turns DSL text into results, tying together
the three building blocks:

* the calculated-metric catalog (PR-D1) -- ``dsl/catalog.py``
* the textual DSL parser / IR (PR-D2) -- ``dsl/parser.py`` + ``dsl/ast.py``
* the name-grounding / synonym resolver (PR-D3) -- ``dsl/grounding.py``

The engine reuses the existing ``query_planner.plan_query`` / ``validate_plan``
/ ``PlanResult`` (PR-5) for scoped missing-table / schema-contract errors and
best-effort degradation, and ``JoinTree`` for fan-out-safe multi-table joins. It
reuses ``dataset_fingerprint`` so repeated identical DSL against unchanged data
is cacheable.

This is the "plan -> execute" integration that ``api-multisite`` does via Cypher
and that ``nlp_api`` *fails* to do (its IR is handed to an opaque external
service). Here the SQL runs on the ``DataSource`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.dsl.ast import DslQuery, metric_token_aliases
from demos.analytics.src.analytics.dsl.catalog import (
    MetricCatalog,
    catalog_for_source,
)
from demos.analytics.src.analytics.dsl.grounding import NameResolver
from demos.analytics.src.analytics.dsl.parser import parse
from demos.analytics.src.analytics.query_planner import (
    PlanResult,
    QuerySpec,
    plan_query,
)


@dataclass
class DslResult:
    """The outcome of running a DSL query through the engine."""

    rows: list[dict[str, Any]]
    sql: str
    spec: QuerySpec
    warnings: list[str] = field(default_factory=list)


class DslEngine:
    """Parse -> ground -> plan -> execute a textual analytics DSL query."""

    def __init__(
        self,
        source: Any,
        model: Any,
        catalog: MetricCatalog | None = None,
        synonyms: NameResolver | dict[str, str] | None = None,
        *,
        best_effort: bool = False,
        catalog_dir: str | Path | None = None,
        base_catalog: MetricCatalog | None = None,
    ) -> None:
        self.source = source
        self.model = model
        self.best_effort = best_effort

        # Catalog: auto-select the dataset-tailored catalog (PR-D1) when a dir is
        # given, otherwise seed one in-memory from the model's base metrics.
        if catalog is None and catalog_dir is not None:
            catalog, _ = catalog_for_source(
                source,
                catalog_dir=catalog_dir,
                base_catalog=base_catalog,
                model=model,
                create=True,
            )
        if catalog is None:
            catalog = MetricCatalog()
            catalog.seed_from_model(model)
        self.catalog = catalog

        # Resolver: a plain synonyms dict is wrapped in a ``NameResolver``.
        if isinstance(synonyms, dict):
            self.resolver: NameResolver = NameResolver.from_dict({"synonyms": synonyms})
        elif synonyms is None:
            self.resolver = NameResolver()
        else:
            self.resolver = synonyms

        # Cache planned SQL per DSL text (only safe when not best-effort, since a
        # dropped table depends on the live source state). Keyed by dataset_sig so
        # the cache is invalidated if the underlying data changes shape.
        self._sig = fingerprint(source, row_count_aware=False)
        self._cache: dict[str, tuple[PlanResult, QuerySpec]] = {}

    # -- core ----------------------------------------------------------------
    def _plan(self, dsl_text: str) -> tuple[PlanResult, QuerySpec, DslQuery]:
        cache_key = f"{self._sig}|{dsl_text}"
        if not self.best_effort and cache_key in self._cache:
            return self._cache[cache_key]
        query: DslQuery = parse(dsl_text)
        spec = query.to_spec(self.model, self.catalog, self.resolver)
        result = plan_query(self.model, spec, self.source, best_effort=self.best_effort)
        if not self.best_effort:
            self._cache[cache_key] = (result, spec, query)
        return result, spec, query

    def query(self, dsl_text: str) -> DslResult:
        """Parse, ground, plan, and execute ``dsl_text`` against the source."""
        result, spec, query = self._plan(dsl_text)
        rows = self.source.native_query(result.sql) if result.sql else []
        rows = _rename_columns(rows, query, self.model, self.catalog, self.resolver)
        return DslResult(rows=rows, sql=result.sql, spec=spec, warnings=result.warnings)

    def explain(self, dsl_text: str) -> str:
        """Return the planned SQL without executing (for agents / debugging)."""
        result, _, _ = self._plan(dsl_text)
        return result.sql


def _rename_columns(
    rows: list[dict[str, Any]],
    query: DslQuery,
    model: Any,
    catalog: MetricCatalog | None,
    resolver: NameResolver,
) -> list[dict[str, Any]]:
    """Rename output columns from source aliases back to the friendly DSL tokens.

    A base metric grounded to ``sales.amount`` selects as ``amount``; rename it
    back to the caller's ``revenue`` so the answer uses business vocabulary.
    """
    # Only rename when the DSL token is a friendly name (no ``table.column`` ref),
    # so a bare ``sales.amount`` keeps its source column name and a synonym like
    # ``revenue`` is restored from ``amount``.
    rename = {
        alias: token
        for token, alias in metric_token_aliases(
            query.metrics, model, catalog, resolver
        ).items()
        if alias != token and "." not in token
    }
    if not rename or not rows:
        return rows
    return [{rename.get(k, k): v for k, v in row.items()} for row in rows]
