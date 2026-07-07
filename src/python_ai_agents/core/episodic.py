"""Episodic memory: cross-session lessons scoped by tenant."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["Episode", "EpisodicStore", "InMemoryEpisodicStore"]


@dataclass(frozen=True, slots=True)
class Episode:
    """A past experience an agent can learn from.

    The tenant scopes recall, so a lesson learned for one tenant is never
    surfaced to another.
    """

    tenant: str
    task: str
    outcome: str
    success: bool
    lesson: str

    def __post_init__(self) -> None:
        if not self.tenant or not self.tenant.strip():
            object.__setattr__(self, "tenant", "default")


class EpisodicStore(Protocol):
    """Stores and recalls episodes, scoped by tenant."""

    async def record(self, episode: Episode) -> None: ...

    async def recall(self, tenant: str, query: str, limit: int) -> list[Episode]: ...


@dataclass
class InMemoryEpisodicStore:
    """Bounded in-memory episodic store for tests and single-process demos."""

    max_entries: int = 10_000
    _entries: OrderedDict[int, Episode] = field(default_factory=OrderedDict, init=False)
    _counter: int = field(default=0, init=False)

    async def record(self, episode: Episode) -> None:
        self._counter += 1
        self._entries[self._counter] = episode
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    async def recall(self, tenant: str, query: str, limit: int) -> list[Episode]:
        results = [
            ep for ep in self._entries.values() if ep.tenant == tenant and _matches(query, ep.task)
        ]
        return results[:limit]


def _matches(query: str, task: str) -> bool:
    query_lower = query.lower()
    task_lower = task.lower()
    if query_lower in task_lower or task_lower in query_lower:
        return True
    query_words = set(query_lower.split())
    task_words = set(task_lower.split())
    return len(query_words & task_words) > 0
