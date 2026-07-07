"""LLM-assisted semantic schema refinement for profiled CSV-style data.

The deterministic profiler discovers types, statistics, samples, relationships,
and first-pass semantic roles. CSVs often need one more pass because column
names such as ``amt``, ``status_code``, or ``flag`` are ambiguous without
business context. This module asks a model to refine the profile into better
semantic roles while keeping the result bounded to discovered tables/columns.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from demos.analytics.src.analytics.data_source import (
    ColumnRole,
    ColumnSchema,
    DataSource,
    TableSchema,
)
from demos.analytics.src.analytics.profiler import DatasetProfile, profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from python_ai_agents import RequestContext, extract_structured
from python_ai_agents.core.model import ModelPort

__all__ = [
    "ColumnSemanticHint",
    "SchemaHints",
    "build_semantic_model",
    "refine_profile_with_llm",
]


class ColumnSemanticHint(BaseModel):
    """One LLM-proposed semantic role refinement for a discovered column."""

    table: str
    column: str
    role: str = Field(description="One ColumnRole value, for example measure_additive")
    description: str = ""


class SchemaHints(BaseModel):
    """LLM output containing role hints for discovered columns only."""

    columns: list[ColumnSemanticHint] = Field(default_factory=list)


async def build_semantic_model(
    source: DataSource,
    model: ModelPort | None = None,
    catalog: Any | None = None,
) -> SemanticModel:
    """Profile a source and build a semantic model, optionally refined by an LLM."""

    profile = profile_dataset(source, catalog)
    if model is not None:
        profile = await refine_profile_with_llm(profile, model)
    return SemanticModel.from_profile(profile)


async def refine_profile_with_llm(
    profile: DatasetProfile,
    model: ModelPort,
    *,
    max_columns: int = 120,
) -> DatasetProfile:
    """Use an LLM to refine deterministic semantic roles.

    The LLM cannot invent columns: hints for unknown tables/columns or invalid
    roles are ignored. On model or validation failure, the original profile is
    returned unchanged.
    """

    prompt = _schema_prompt(profile, max_columns)
    try:
        result = await extract_structured(
            model,
            SchemaHints,
            prompt,
            RequestContext.ephemeral(),
            max_retries=2,
            system_prompt=(
                "You refine semantic roles for CSV analytics schemas. "
                "Use only discovered tables and columns. Prefer evidence from "
                "types, sample values, statistics, and column names. Do not invent columns."
            ),
        )
    except Exception:
        return profile
    if result.value is None:
        return profile
    return _apply_hints(profile, result.value)


def _schema_prompt(profile: DatasetProfile, max_columns: int) -> str:
    payload = {
        "allowedRoles": [role.value for role in ColumnRole],
        "guidance": [
            "Use identifier only for true entity keys, foreign keys, codes, UUIDs, or IDs.",
            "Use measure_additive for amounts, counts, balances, quantities, "
            "revenue, cost, and numeric facts that can be summed.",
            "Use measure_ratio for rates, percentages, proportions, "
            "scores bounded to 0-1 or 0-100, and averages.",
            "Use date or timestamp for temporal columns.",
            "Use dimension for categorical grouping columns.",
            "Use text for free-form natural language fields.",
        ],
        "tables": [{"name": table.name, "rows": table.rows} for table in profile.tables],
        "columns": [_column_payload(profile, column) for column in profile.columns[:max_columns]],
        "relationships": [
            {
                "from": f"{rel.from_table}.{','.join(rel.from_columns)}",
                "to": f"{rel.to_table}.{','.join(rel.to_columns)}",
                "cardinality": rel.cardinality,
                "coverage": rel.coverage,
            }
            for rel in profile.relationships
        ],
    }
    return (
        "Refine the semantic role for each column where the deterministic role looks wrong. "
        "Return hints for every column you are confident about, "
        "using the exact table and column names.\n\n"
        f"Profile:\n{json.dumps(payload, default=str, indent=2)}"
    )


def _column_payload(profile: DatasetProfile, column) -> dict[str, Any]:
    table = _table_for(profile, column.table)
    deterministic_role = ""
    if table is not None:
        deterministic_role = next(
            (col.role.value for col in table.columns if col.name == column.name),
            "",
        )
    return {
        "table": column.table,
        "name": column.name,
        "physicalType": column.physical_type,
        "deterministicRole": deterministic_role,
        "rows": column.rows,
        "distinct": column.distinct,
        "nulls": column.nulls,
        "min": column.min,
        "max": column.max,
        "mean": column.mean,
        "signals": sorted(column.signals),
        "sampleValues": list(column.sample_values[:10]),
    }


def _apply_hints(profile: DatasetProfile, hints: SchemaHints) -> DatasetProfile:
    known_refs = {(table.name, column.name) for table in profile.tables for column in table.columns}
    role_overrides: dict[tuple[str, str], ColumnRole] = {}
    for hint in hints.columns:
        key = (hint.table, hint.column)
        if key not in known_refs:
            continue
        role = _parse_role(hint.role)
        if role is not None:
            role_overrides[key] = role

    if not role_overrides:
        return profile

    tables: list[TableSchema] = []
    for table in profile.tables:
        columns = []
        for column in table.columns:
            role = role_overrides.get((table.name, column.name), column.role)
            columns.append(
                ColumnSchema(
                    name=column.name,
                    physical_type=column.physical_type,
                    role=role,
                )
            )
        tables.append(TableSchema(name=table.name, rows=table.rows, columns=tuple(columns)))

    return DatasetProfile(
        tables=tuple(tables),
        columns=profile.columns,
        relationships=profile.relationships,
        import_plan=profile.import_plan,
    )


def _parse_role(value: str) -> ColumnRole | None:
    normalized = value.strip().lower()
    for role in ColumnRole:
        if role.value == normalized or role.name.lower() == normalized:
            return role
    return None


def _table_for(profile: DatasetProfile, name: str):
    return next((table for table in profile.tables if table.name == name), None)
