"""Calculated-metric catalog for the analytics DSL engine (PR-D1).

A ``MetricCatalog`` is the semantic layer: it holds *base* metrics (proxies to a
source column + aggregation) and *calculated* metrics (named expressions that
may reference other metrics, including other calculated metrics). It expands a
calculated metric to additive-safe aggregated SQL so ratios stay correct under
grouping, validates dependencies (cycle detection + base-column existence), and
persists **per dataset** under a ``dataset_sig`` key (see ``catalog_for_source``)
so the same generic engine auto-tailors to health / casino / retail data with no
domain branching.

This deliberately reuses the existing engine rather than re-implementing it:
``SemanticModel`` (for base metrics / validation), ``dataset_fingerprint`` (for
the per-dataset key, row-count-agnostic and consistent with PR-11), and
``file_lock`` (atomic, cross-process-safe persistence).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demos.analytics.src.analytics.data_source import sql_qcol
from demos.analytics.src.analytics.dataset_fingerprint import fingerprint
from demos.analytics.src.analytics.file_lock import atomic_write_text, file_lock
from demos.analytics.src.analytics.semantic_model import SemanticModel

_CATALOG_DIR_ENV = "ANALYTICS_DSL_CATALOG_DIR"


class CatalogError(ValueError):
    """A scoped failure in the metric catalog: cycle, unknown ref, bad column."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"metric catalog error for '{name}': {reason}")


@dataclass(frozen=True, slots=True)
class MetricDef:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class BaseMetricDef(MetricDef):
    table: str
    column: str
    aggregation: str = "sum"


@dataclass(frozen=True, slots=True)
class CalculatedMetricDef(MetricDef):
    expression: str
    aggregation_hint: str | None = None


# A token is a bare identifier, or ident.ident (a ``table.column`` source ref).
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|)?")


class MetricCatalog:
    """Named metric registry: base (source) + calculated (expression) metrics."""

    def __init__(self, defs: list[MetricDef] | None = None) -> None:
        self._defs: dict[str, MetricDef] = {}
        for d in defs or ():
            self.add(d)

    # -- mutation ----------------------------------------------------------
    def add(self, def_: MetricDef) -> None:
        self._defs[def_.name.lower()] = def_

    def get(self, name: str) -> MetricDef | None:
        return self._defs.get(name.lower())

    def names(self) -> list[str]:
        return sorted(self._defs)

    def seed_from_model(self, model: SemanticModel) -> MetricCatalog:
        """Add a ``BaseMetricDef`` for every model metric so they are discoverable.

        Existing definitions (e.g. a calculated metric) are never overwritten.
        """
        for m in model.metrics:
            if self.get(m.ref) is None:
                self.add(
                    BaseMetricDef(
                        name=m.ref,
                        table=m.table,
                        column=m.column,
                        aggregation=m.aggregation,
                        description="",
                    )
                )
        return self

    # -- resolution --------------------------------------------------------
    def resolve(self, name: str, model: SemanticModel) -> tuple[str, str]:
        """Return ``(aggregated_sql_expr, alias)`` for a metric name or ref."""
        return self._resolve(name, model, set())

    def _resolve(self, name: str, model: SemanticModel, seen: set[str]) -> tuple[str, str]:
        if name.lower() in seen:
            raise CatalogError(name, "cyclic dependency among calculated metrics")
        def_ = self.get(name)
        if def_ is None:
            base = self._base_from_model(model, name)
            if base is None:
                raise CatalogError(name, "unknown metric (not in catalog or source model)")
            return (
                f"{base.aggregation.upper()}({sql_qcol(base.table, base.column)})",
                base.column,
            )
        if isinstance(def_, BaseMetricDef):
            return (
                f"{def_.aggregation.upper()}({sql_qcol(def_.table, def_.column)})",
                def_.name,
            )
        # Calculated: expand its expression, substituting referenced metrics with
        # their *aggregated* SQL so ratios stay additive-safe under GROUP BY.
        sql = self._expand(def_.expression, model, seen | {name.lower()})
        return (f"({sql})", def_.name)

    @staticmethod
    def _base_from_model(model: SemanticModel, name: str) -> Any | None:
        nl = name.lower()
        for m in model.metrics:
            if m.ref.lower() == nl or m.column.lower() == nl:
                return m
        return None

    def _expand(self, expr: str, model: SemanticModel, seen: set[str]) -> str:
        # Only resolve the catalog names that actually appear in the expression.
        tokens = {m.group(0).lower() for m in _TOKEN.finditer(expr) if "." not in m.group(0)}
        cat_sql: dict[str, str] = {}
        for tok in tokens:
            if self.get(tok) is not None:
                cat_sql[tok] = self._resolve(tok, model, seen)[0]

        base_sql: dict[str, str] = {}
        for m in model.metrics:
            s = f"{m.aggregation.upper()}({sql_qcol(m.table, m.column)})"
            base_sql[m.ref.lower()] = s
            base_sql[m.column.lower()] = s

        def repl(match: re.Match[str]) -> str:
            tok = match.group(0)
            tl = tok.lower()
            if "." in tok:
                return base_sql.get(tl, tok)  # dotted -> source ref, else leave
            if tl in cat_sql:
                return cat_sql[tl]
            return tok  # SQL keyword / function / literal

        return _TOKEN.sub(repl, expr)

    # -- validation --------------------------------------------------------
    def validate(self, model: SemanticModel) -> None:
        """Fail fast on cycles or references to non-existent base columns."""
        for name, def_ in list(self._defs.items()):
            if isinstance(def_, CalculatedMetricDef):
                self._check_deps(name, def_.expression, model, set())

    def _check_deps(self, name: str, expr: str, model: SemanticModel, seen: set[str]) -> None:
        if name.lower() in seen:
            raise CatalogError(name, "cyclic dependency among calculated metrics")
        seen = seen | {name.lower()}
        for m in _TOKEN.finditer(expr):
            tok = m.group(0)
            tl = tok.lower()
            if "." in tok:
                if self._base_from_model(model, tok) is None:
                    raise CatalogError(name, f"expression references unknown base column '{tok}'")
            else:
                ref = self.get(tl)
                if ref is None:
                    continue  # not a catalog name -> SQL keyword/function/literal
                if isinstance(ref, CalculatedMetricDef):
                    self._check_deps(ref.name, ref.expression, model, seen)

    # -- layering ----------------------------------------------------------
    def override(self, other: MetricCatalog) -> MetricCatalog:
        """Return a new catalog with ``other``'s definitions taking precedence.

        Used to layer a dataset-specific catalog over a shared base catalog, and
        then a per-customer override on top (mirrors nlp_api's catalog layering).
        """
        merged = MetricCatalog()
        for d in self._defs.values():
            merged.add(d)
        for d in other._defs.values():
            merged.add(d)
        return merged

    merge = override  # alias

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        metrics: list[dict[str, Any]] = []
        for d in self._defs.values():
            if isinstance(d, BaseMetricDef):
                metrics.append(
                    {
                        "kind": "base",
                        "name": d.name,
                        "table": d.table,
                        "column": d.column,
                        "aggregation": d.aggregation,
                        "description": d.description,
                    }
                )
            else:
                metrics.append(
                    {
                        "kind": "calculated",
                        "name": d.name,
                        "expression": d.expression,
                        "aggregation_hint": d.aggregation_hint,
                        "description": d.description,
                    }
                )
        return {"metrics": metrics}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricCatalog:
        cat = cls()
        for item in data.get("metrics", []):
            if item.get("kind") == "base":
                cat.add(
                    BaseMetricDef(
                        name=item["name"],
                        table=item["table"],
                        column=item["column"],
                        aggregation=item.get("aggregation", "sum"),
                        description=item.get("description", ""),
                    )
                )
            else:
                cat.add(
                    CalculatedMetricDef(
                        name=item["name"],
                        expression=item["expression"],
                        aggregation_hint=item.get("aggregation_hint"),
                        description=item.get("description", ""),
                    )
                )
        return cat


class CatalogStore:
    """Persists one ``MetricCatalog`` per dataset, keyed by its ``dataset_sig``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_sig: str) -> Path:
        return self.directory / f"{dataset_sig}.json"

    def load(self, dataset_sig: str) -> MetricCatalog:
        p = self.path_for(dataset_sig)
        if p.exists():
            return MetricCatalog.from_dict(json.loads(p.read_text()))
        return MetricCatalog()

    def save(self, dataset_sig: str, catalog: MetricCatalog) -> None:
        p = self.path_for(dataset_sig)
        with file_lock(p):
            atomic_write_text(p, json.dumps(catalog.to_dict(), indent=2, sort_keys=True))


def catalog_for_source(
    source: Any,
    *,
    catalog_dir: str | Path | None = None,
    base_catalog: MetricCatalog | None = None,
    model: SemanticModel | None = None,
    create: bool = True,
) -> tuple[MetricCatalog, str]:
    """Resolve (and optionally create) the dataset-tailored catalog for ``source``.

    The catalog file is keyed by ``dataset_sig`` — the row-count-agnostic
    fingerprint (PR-11) — so loading health data selects the health-tailored
    catalog, loading casino selects the casino one, with zero engine branching.

    Selection order (each layer overrides the previous on name conflict):
      1. the dataset-specific catalog ``<dataset_sig>.json`` (if present);
      2. merged over an optional shared ``base_catalog`` (cross-domain metrics);
      3. seeded in-memory with the ``SemanticModel`` base metrics for discovery.

    Base metrics are always resolvable by ``table.column`` ref even with no
    tailored catalog; the tailored catalog only *adds* calculated metrics.
    """
    sig = fingerprint(source, row_count_aware=False)
    store = CatalogStore(catalog_dir or os.environ.get(_CATALOG_DIR_ENV, "./dsl_catalogs"))

    dataset_cat = store.load(sig)
    if model is not None:
        dataset_cat.validate(model)  # fail fast on broken dataset-specific defs

    effective = dataset_cat
    if base_catalog is not None:
        effective = base_catalog.override(dataset_cat)  # dataset wins

    if model is not None:
        seeded = MetricCatalog()
        for d in effective._defs.values():
            seeded.add(d)
        seeded.seed_from_model(model)
        effective = seeded

    if create and not store.path_for(sig).exists():
        store.save(sig, dataset_cat)

    return effective, sig
