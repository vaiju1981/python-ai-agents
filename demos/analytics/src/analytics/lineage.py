"""Cross-answer lineage graph (PR-7, tracker §G G4).

Provenance envelopes are per-answer: each carries the SQL and ``dataset_sig`` it
was produced from, but a *derived* answer (a forecast built on a reconciled
metric, a model scored on profiled data) cannot be traced back through every
upstream answer. This module keeps a small directed graph of answers:

    dataset_sig -> SQL -> answer id -> downstream answer id

so ``trace_lineage(answer_id)`` walks a derived answer all the way back to its
raw sources. It is file-backed (``lineage.json``) so it persists alongside the
``audit_store`` and survives restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class LineageNode:
    """One recorded answer in the lineage graph."""

    answer_id: str
    dataset_sig: str
    sql: str
    parents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answerId": self.answer_id,
            "datasetSig": self.dataset_sig,
            "sql": self.sql,
            "parents": list(self.parents),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LineageNode:
        return cls(
            answer_id=d["answerId"],
            dataset_sig=d.get("datasetSig", ""),
            sql=d.get("sql", ""),
            parents=list(d.get("parents", [])),
        )


class LineageGraph:
    """File-backed graph of answers and their upstream dependencies.

    ``record`` adds a node (an answer id, its ``dataset_sig`` + ``sql``, and the
    answer ids it was built from). ``trace_lineage`` walks a node's parents up to
    the raw sources, returning the chain in dependency order.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._nodes: dict[str, LineageNode] = {}
        # Conversation-scoped list of the most recent answer ids. Tools append
        # their own id here after recording, so a later answer in the same
        # conversation links back to the answers produced before it. Shared across
        # the descriptive + predictive toolsets because they hold the same graph.
        self.scope: list[str] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return
        for n in data.get("nodes", []):
            try:
                self._nodes[n["answerId"]] = LineageNode.from_dict(n)
            except Exception:
                continue

    def _persist(self) -> None:
        if self.path is None:
            return
        try:
            self.path.write_text(
                json.dumps(
                    {"nodes": [n.to_dict() for n in self._nodes.values()]}, indent=2
                )
            )
        except Exception:
            return None

    @staticmethod
    def new_id() -> str:
        """Allocate a fresh answer id for a tool result."""
        return uuid4().hex

    def record(
        self,
        answer_id: str,
        dataset_sig: str,
        sql: str,
        parents: list[str] | None = None,
    ) -> LineageNode:
        node = LineageNode(
            answer_id=answer_id,
            dataset_sig=dataset_sig,
            sql=sql,
            parents=list(parents or []),
        )
        self._nodes[answer_id] = node
        self._persist()
        return node

    def node(self, answer_id: str) -> LineageNode | None:
        return self._nodes.get(answer_id)

    def trace_lineage(self, answer_id: str) -> list[LineageNode]:
        """Walk ``answer_id``'s parents upstream to raw sources.

        Returns the chain in dependency order: the answer itself first, then each
        upstream answer it was built from (depth-first), ending at the roots.
        """
        out: list[LineageNode] = []
        seen: set[str] = set()

        def visit(aid: str) -> None:
            if aid in seen:
                return
            node = self._nodes.get(aid)
            if node is None:
                return
            seen.add(aid)
            out.append(node)
            for parent in node.parents:
                visit(parent)

        visit(answer_id)
        return out

    def upstream_dataset_sigs(self, answer_id: str) -> set[str]:
        """Distinct dataset signatures across the answer and everything upstream."""
        return {n.dataset_sig for n in self.trace_lineage(answer_id) if n.dataset_sig}
