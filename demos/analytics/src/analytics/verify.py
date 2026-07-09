"""Semantic verification of analytics requests (cross-cut defensibility).

Before a question is answered, verify that the requested metrics / dimensions /
filters actually exist in the semantic model and are *answerable* — i.e. the
metric is a real measure (not a free-text column being summed), the dimensions
are groupable, and the filters reference known columns. This catches the LLM
choosing the wrong column or an unanswerable aggregation before it produces a
confident-but-wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from demos.analytics.src.analytics.semantic_model import SemanticModel


@dataclass
class Verification:
    ok: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "issues": self.issues, "suggestions": self.suggestions}


def verify(
    model: SemanticModel,
    metrics: list[str] | None = None,
    dimensions: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> Verification:
    """Validate an analytics request against the semantic model."""
    issues: list[str] = []
    suggestions: list[str] = []
    metrics = metrics or []
    dimensions = dimensions or []
    filters = filters or []

    metric_refs = {m.ref.lower(): m for m in model.metrics}
    metric_cols = {m.column.lower(): m for m in model.metrics}
    dim_refs = {d.ref.lower(): d for d in model.dimensions}
    dim_cols = {d.column.lower(): d for d in model.dimensions}
    col_by_ref = {c.ref.lower(): c for c in model.columns}
    col_by_name = {c.name.lower(): c for c in model.columns}
    all_cols = {c.lower() for m in model.metrics for c in (m.ref, m.column)}
    all_cols |= {c.lower() for d in model.dimensions for c in (d.ref, d.column)}
    all_cols |= {c.ref.lower() for c in model.columns}
    all_cols |= {c.name.lower() for c in model.columns}

    # Metrics must exist and be aggregatable measures.
    for ref in metrics:
        rl = ref.lower()
        m = metric_refs.get(rl) or metric_cols.get(rl)
        if m is None:
            # Distinguish "column exists but isn't a measure" from "unknown".
            col = col_by_ref.get(rl) or col_by_name.get(rl)
            if col is not None:
                issues.append(
                    f"metric '{ref}' is a {col.role.value} column; "
                    "cannot be aggregated as a measure"
                )
                suggestions.append("choose a numeric measure column to aggregate")
            else:
                issues.append(f"unknown metric '{ref}'")
                suggestions.append(f"did you mean one of: {sorted(all_cols)[:5]}")
            continue
        if m.role.value in ("text", "boolean"):
            issues.append(
                f"metric '{ref}' is a {m.role.value} column; cannot be aggregated as a measure"
            )

    # Dimensions must exist and be groupable (not high-cardinality measures).
    for ref in dimensions:
        rl = ref.lower()
        d = dim_refs.get(rl) or dim_cols.get(rl)
        if d is None:
            issues.append(f"unknown dimension '{ref}'")
            continue
        if d.role.value in ("measure_additive", "measure_ratio"):
            issues.append(f"dimension '{ref}' is a measure; grouping by it is unusual")
            suggestions.append("use a category/identifier column to group")

    # Filters must reference known columns.
    for f in filters:
        col = f.get("column", "")
        if "." in col:
            col = col.split(".", 1)[1]
        if col.lower() not in all_cols:
            issues.append(f"filter references unknown column '{col}'")

    return Verification(ok=len(issues) == 0, issues=issues, suggestions=suggestions)
