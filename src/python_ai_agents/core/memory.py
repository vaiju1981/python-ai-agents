from __future__ import annotations

from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncContextManager, AsyncIterator, Callable, Protocol

import anyio

from python_ai_agents.core.model import Message, Role


class Memory(Protocol):
    def add(self, message: Message) -> None:
        ...

    def history(self) -> tuple[Message, ...]:
        ...


class InMemoryMemory:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def history(self) -> tuple[Message, ...]:
        return tuple(self._messages)


class WindowedMemory:
    """Keeps system messages plus the most recent non-system messages."""

    def __init__(self, max_recent: int = 32) -> None:
        self.max_recent = max(1, max_recent)
        self._system_messages: list[Message] = []
        self._recent: deque[Message] = deque()

    def add(self, message: Message) -> None:
        if message.role == Role.SYSTEM:
            self._system_messages.append(message)
            return
        self._recent.append(message)
        while len(self._recent) > self.max_recent:
            self._recent.popleft()

    def history(self) -> tuple[Message, ...]:
        return (*self._system_messages, *tuple(self._recent))


class ConversationStore(Protocol):
    def memory(self, tenant: str, session_id: str) -> AsyncContextManager[Memory]:
        ...


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    message_count: int
    last_activity: datetime


class ConversationHistory(Protocol):
    def list_sessions(self, tenant: str) -> list[SessionSummary]:
        ...

    def messages(self, tenant: str, session_id: str) -> tuple[Message, ...]:
        ...

    def delete(self, tenant: str, session_id: str) -> None:
        ...


@dataclass(slots=True)
class _Entry:
    memory: Memory
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    pins: int = 0
    last_access: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.fromtimestamp(0, timezone.utc))


class InMemoryConversationStore:
    """Bounded, tenant/session-scoped conversation memory."""

    def __init__(
        self,
        memory_factory: Callable[[], Memory] | None = None,
        max_sessions: int = 10_000,
    ) -> None:
        self.memory_factory = memory_factory or InMemoryMemory
        self.max_sessions = max(1, max_sessions)
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._lock = anyio.Lock()
        self._clock = 0

    @asynccontextmanager
    async def memory(self, tenant: str, session_id: str) -> AsyncIterator[Memory]:
        key = (tenant, session_id)
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(self.memory_factory())
                self._entries[key] = entry
            self._clock += 1
            entry.pins += 1
            entry.last_access = self._clock
            entry.last_activity = datetime.now(timezone.utc)

        try:
            async with entry.lock:
                yield entry.memory
        finally:
            async with self._lock:
                current = self._entries.get(key)
                if current is not None:
                    current.pins = max(0, current.pins - 1)
                self._evict_if_needed()

    def list_sessions(self, tenant: str) -> list[SessionSummary]:
        sessions = [
            SessionSummary(
                session_id=session_id,
                message_count=len(entry.memory.history()),
                last_activity=entry.last_activity,
            )
            for (entry_tenant, session_id), entry in self._entries.items()
            if entry_tenant == tenant
        ]
        return sorted(sessions, key=lambda session: session.last_activity, reverse=True)

    def messages(self, tenant: str, session_id: str) -> tuple[Message, ...]:
        entry = self._entries.get((tenant, session_id))
        if entry is None:
            return ()
        return entry.memory.history()

    def delete(self, tenant: str, session_id: str) -> None:
        self._entries.pop((tenant, session_id), None)

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_sessions:
            victim = next(
                (
                    key
                    for key, entry in sorted(
                        self._entries.items(),
                        key=lambda item: item[1].last_access,
                    )
                    if entry.pins == 0
                ),
                None,
            )
            if victim is None:
                return
            self._entries.pop(victim, None)
