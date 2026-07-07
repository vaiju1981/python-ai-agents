"""User-editable metadata: include/exclude relationship overrides and descriptions.

Persisted as one JSON file so a UI can edit it live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from demos.analytics.src.analytics.data_source import Relationship


@dataclass
class Catalog:
    """User-editable metadata layered on top of auto-discovered relationships."""

    file: Path | None = None
    include: list[dict[str, Any]] = field(default_factory=list)
    exclude: list[dict[str, Any]] = field(default_factory=list)
    table_descriptions: dict[str, str] = field(default_factory=dict)
    column_descriptions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, file: Path | None = None) -> Catalog:
        if file is None or not file.exists():
            return cls(file=file)
        raw = json.loads(file.read_text())
        rel = raw.get("relationships", {})
        desc = raw.get("descriptions", {})
        return cls(
            file=file,
            include=rel.get("include", []),
            exclude=rel.get("exclude", []),
            table_descriptions=desc.get("tables", {}),
            column_descriptions=desc.get("columns", {}),
        )

    def apply(self, discovered: list[Relationship]) -> list[Relationship]:
        """Apply include/exclude overrides to discovered relationships."""
        out: list[Relationship] = []
        for r in discovered:
            excluded = any(_rel_match(x, r) for x in self.exclude)
            if not excluded:
                out.append(r)
        for inc in self.include:
            out.append(
                Relationship(
                    from_table=inc["fromTable"],
                    from_columns=tuple(inc.get("fromColumns", [])),
                    to_table=inc["toTable"],
                    to_columns=tuple(inc.get("toColumns", [])),
                    cardinality=inc.get("cardinality", "many_to_one"),
                    coverage=1.0,
                )
            )
        return out

    def description_for_table(self, name: str) -> str:
        return self.table_descriptions.get(name, "")

    def description_for_column(self, table: str, column: str) -> str | None:
        key = f"{table}.{column}"
        return self.column_descriptions.get(key)

    def save(self) -> None:
        if self.file is None:
            return
        data = {
            "relationships": {"include": self.include, "exclude": self.exclude},
            "descriptions": {
                "tables": self.table_descriptions,
                "columns": self.column_descriptions,
            },
        }
        self.file.write_text(json.dumps(data, indent=2))


def _rel_match(exclude: dict[str, Any], r: Relationship) -> bool:
    return (
        exclude.get("fromTable") == r.from_table
        and exclude.get("toTable") == r.to_table
        and tuple(exclude.get("fromColumns", [])) == r.from_columns
        and tuple(exclude.get("toColumns", [])) == r.to_columns
    )
