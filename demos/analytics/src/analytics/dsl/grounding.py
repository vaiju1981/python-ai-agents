"""Name grounding / synonym resolver for the analytics DSL (PR-D3).

Callers (and the NL bridge, PR-D5) use business vocabulary — ``revenue``,
``region``, ``last 30 days`` — not ``sales.amount`` / ``sales.region``. This
module grounds those friendly names to the engine's internal refs and catalog
metric names, with pluggable synonym sources (built-in model columns, a JSON
synonyms file, per-customer overrides) and an optional fuzzy fallback
(edit-distance) behind a flag.

This mirrors ``nlp_api``'s ``NamingConventions`` triple-map and
``VALUE_SYNONYMS`` (and its local edit-distance string matching) but with no
Dialogflow dependency and no opaque external handoff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.dsl.catalog import (
    BaseMetricDef,
    MetricCatalog,
    MetricDef,
)
from demos.analytics.src.analytics.semantic_model import (
    Dimension,
    Metric,
    SemanticModel,
)


class UnresolvedNameError(ValueError):
    """A scoped failure grounding a friendly name; names the kind of name."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind
        super().__init__(f"could not resolve {kind} '{name}'")


def _norm(token: str) -> str:
    """Lower-case and collapse internal whitespace so ``NeT  WiN`` == ``net win``."""
    return re.sub(r"\s+", " ", token.lower()).strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class NameResolver:
    """Layered name resolver: ref -> catalog name -> synonym -> (optional) fuzzy.

    Synonym sources (lowercased keys/values):
      * ``synonyms``: friendly token -> metric/dimension ref or catalog name.
      * ``value_synonyms``: dimension ref -> {friendly value -> normalized value}
        (mirrors ``nlp_api``'s ``DIMENSION_VALUES_DETECTION_MAPPER``).
      * ``period_synonyms``: friendly period phrase -> normalized phrase the DSL
        parser already understands (e.g. ``last 30 days`` -> ``30 days``).
    Per-customer overrides are supplied by constructing a resolver with a merged
    synonym map (same layering as ``nlp_api`` settings).
    """

    synonyms: dict[str, str] = field(default_factory=dict)
    value_synonyms: dict[str, dict[str, str]] = field(default_factory=dict)
    period_synonyms: dict[str, str] = field(default_factory=dict)
    fuzzy: bool = False
    fuzzy_budget: int = 2

    def __post_init__(self) -> None:
        # Normalize synonym keys/values to lower case for case-insensitive lookup.
        self.synonyms = {k.lower(): v for k, v in self.synonyms.items()}
        self.value_synonyms = {
            k: {vk.lower(): vv for vk, vv in v.items()}
            for k, v in self.value_synonyms.items()
        }
        self.period_synonyms = {k.lower(): v for k, v in self.period_synonyms.items()}

    # -- metrics -------------------------------------------------------------
    @staticmethod
    def _base_def(m: Metric) -> BaseMetricDef:
        return BaseMetricDef(
            name=m.ref,
            table=m.table,
            column=m.column,
            aggregation=m.aggregation,
            description="",
        )

    def resolve_metric(
        self, token: str, catalog: MetricCatalog | None, model: SemanticModel
    ) -> MetricDef:
        tl = _norm(token)

        # 1. Exact source ref or bare column.
        for m in model.metrics:
            if m.ref.lower() == tl or m.column.lower() == tl:
                return self._base_def(m)

        # 2. Catalog name (calculated or previously-seeded base metric).
        if catalog is not None:
            defn = catalog.get(tl)
            if defn is not None:
                return defn

        # 3. Synonym map (friendly name -> ref / catalog name).
        ref = self.synonyms.get(tl)
        if ref is not None:
            return self._def_from_ref(ref, catalog, model, token)

        # 4. Fuzzy fallback within a distance budget.
        if self.fuzzy:
            guess = self._fuzzy_guess(
                tl,
                [m.ref for m in model.metrics]
                + ([c.name for c in (catalog.names() if catalog else [])] if catalog else [])
                + list(self.synonyms.keys()),
            )
            if guess is not None:
                return self.resolve_metric(guess, catalog, model)

        raise UnresolvedNameError(token, "metric")

    def _def_from_ref(
        self, ref: str, catalog: MetricCatalog | None, model: SemanticModel, token: str
    ) -> MetricDef:
        if catalog is not None:
            defn = catalog.get(ref)
            if defn is not None:
                return defn
        for m in model.metrics:
            if m.ref.lower() == ref.lower() or m.column.lower() == ref.lower():
                return self._base_def(m)
        raise UnresolvedNameError(token, "metric")

    # -- dimensions ----------------------------------------------------------
    def resolve_dimension(self, token: str, model: SemanticModel) -> Dimension:
        tl = _norm(token)
        for d in model.dimensions:
            if d.ref.lower() == tl or d.column.lower() == tl:
                return d
        ref = self.synonyms.get(tl)
        if ref is not None:
            for d in model.dimensions:
                if d.ref.lower() == ref.lower() or d.column.lower() == ref.lower():
                    return d
        if self.fuzzy:
            guess = self._fuzzy_guess(
                tl, [d.ref for d in model.dimensions] + list(self.synonyms.keys())
            )
            if guess is not None:
                return self.resolve_dimension(guess, model)
        raise UnresolvedNameError(token, "dimension")

    # -- filter values -------------------------------------------------------
    def resolve_value(self, dimension_ref: str, value: Any) -> Any:
        """Normalize a filter literal via the per-dimension value map."""
        if not isinstance(value, str):
            return value
        vmap = self.value_synonyms.get(dimension_ref)
        if vmap:
            norm = vmap.get(value.lower())
            if norm is not None:
                return norm
        return value

    # -- periods -------------------------------------------------------------
    def resolve_period(self, token: str) -> str:
        """Normalize a period phrase to a form the DSL parser understands.

        Returns the original token if no synonym applies (the parser already
        handles ``30 days``, ``last month`` style phrases directly).
        """
        return self.period_synonyms.get(token.lower().strip(), token)

    # -- helpers -------------------------------------------------------------
    def _fuzzy_guess(self, token: str, candidates: list[str]) -> str | None:
        best, best_d = None, self.fuzzy_budget + 1
        for c in candidates:
            d = _levenshtein(token.lower(), c.lower())
            if d < best_d:
                best, best_d = c, d
        return best

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        base: NameResolver | None = None,
        *,
        fuzzy: bool = False,
        fuzzy_budget: int = 2,
    ) -> NameResolver:
        """Build a resolver from a synonyms mapping, layering over ``base``.

        ``data`` may contain ``synonyms``, ``value_synonyms`` and
        ``period_synonyms`` keys.
        """
        synonyms = dict(base.synonyms if base else {})
        synonyms.update({k.lower(): v for k, v in data.get("synonyms", {}).items()})
        value_synonyms = {
            k: {vk.lower(): vv for vk, vv in v.items()}
            for k, v in data.get("value_synonyms", {}).items()
        }
        period_synonyms = {
            k.lower(): v for k, v in data.get("period_synonyms", {}).items()
        }
        return cls(
            synonyms=synonyms,
            value_synonyms=value_synonyms,
            period_synonyms=period_synonyms,
            fuzzy=fuzzy or (base.fuzzy if base else False),
            fuzzy_budget=fuzzy_budget,
        )
