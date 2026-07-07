from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import anyio

from python_ai_agents.core.model import Message, Role, ToolCall


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
    def memory(self, tenant: str, session_id: str) -> AbstractAsyncContextManager[Memory]:
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


class SQLiteConversationStore:
    """SQLite-backed conversation store for local durable rollouts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._locks: dict[tuple[str, str], anyio.Lock] = {}
        self._locks_lock = anyio.Lock()
        self._initialize()

    @asynccontextmanager
    async def memory(self, tenant: str, session_id: str) -> AsyncIterator[Memory]:
        key = (tenant, session_id)
        async with self._locks_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = anyio.Lock()
                self._locks[key] = lock

        async with lock:
            memory = InMemoryMemory()
            for message in await anyio.to_thread.run_sync(self.messages, tenant, session_id):
                memory.add(message)
            before = len(memory.history())
            yield memory
            after_messages = memory.history()[before:]
            if after_messages:
                await anyio.to_thread.run_sync(
                    self._append_messages,
                    tenant,
                    session_id,
                    after_messages,
                )
            else:
                await anyio.to_thread.run_sync(self._touch_session, tenant, session_id)

    def list_sessions(self, tenant: str) -> list[SessionSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, COUNT(*) AS message_count, MAX(created_at) AS last_activity
                FROM conversation_messages
                WHERE tenant = ?
                GROUP BY session_id
                ORDER BY last_activity DESC
                """,
                (tenant,),
            ).fetchall()
        return [
            SessionSummary(
                session_id=row[0],
                message_count=int(row[1]),
                last_activity=datetime.fromisoformat(row[2]),
            )
            for row in rows
        ]

    def messages(self, tenant: str, session_id: str) -> tuple[Message, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, tool_calls_json, tool_call_id, tool_name
                FROM conversation_messages
                WHERE tenant = ? AND session_id = ?
                ORDER BY ordinal ASC
                """,
                (tenant, session_id),
            ).fetchall()
        return tuple(_message_from_row(row) for row in rows)

    def delete(self, tenant: str, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation_messages WHERE tenant = ? AND session_id = ?",
                (tenant, session_id),
            )

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    tenant TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL,
                    tool_call_id TEXT,
                    tool_name TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant, session_id, ordinal)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
                ON conversation_messages(tenant, session_id, ordinal)
                """
            )

    def _append_messages(
        self,
        tenant: str,
        session_id: str,
        messages: tuple[Message, ...],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(ordinal), -1)
                FROM conversation_messages
                WHERE tenant = ? AND session_id = ?
                """,
                (tenant, session_id),
            ).fetchone()
            next_ordinal = int(row[0]) + 1
            conn.executemany(
                """
                INSERT INTO conversation_messages (
                    tenant, session_id, ordinal, role, content, tool_calls_json,
                    tool_call_id, tool_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        tenant,
                        session_id,
                        next_ordinal + index,
                        message.role.value,
                        message.content,
                        json.dumps([_tool_call_to_json(call) for call in message.tool_calls]),
                        message.tool_call_id,
                        message.tool_name,
                        now,
                    )
                    for index, message in enumerate(messages)
                ],
            )

    def _touch_session(self, tenant: str, session_id: str) -> None:
        # No-op placeholder for stores that track session metadata separately.
        return None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


def _tool_call_to_json(call: ToolCall) -> dict[str, object]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _tool_call_from_json(value: object) -> ToolCall:
    if not isinstance(value, dict):
        raise ValueError("tool call payload must be an object")
    name = value.get("name")
    if not isinstance(name, str):
        raise ValueError("tool call payload is missing a name")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be an object")
    call_id = value.get("id")
    return ToolCall(name=name, arguments=arguments, id=call_id if isinstance(call_id, str) else "")


def _message_from_row(row) -> Message:
    role = Role(row[0])
    content = row[1]
    tool_calls_payload = json.loads(row[2])
    if not isinstance(tool_calls_payload, list):
        raise ValueError("stored tool calls payload must be a list")
    return Message(
        role=role,
        content=content,
        tool_calls=tuple(_tool_call_from_json(call) for call in tool_calls_payload),
        tool_call_id=row[3],
        tool_name=row[4],
    )
