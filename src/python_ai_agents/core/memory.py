from __future__ import annotations

import json
import sqlite3
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncContextManager, AsyncIterator, Callable, Protocol

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


# ---------------------------------------------------------------------------
# Token-windowed and summarizing memory
# ---------------------------------------------------------------------------


class Summarizer(Protocol):
    """Condenses a span of conversation into a short summary.

    Implementations typically call a model; tests supply a deterministic stub.
    """

    def summarize(self, messages: tuple[Message, ...]) -> str:
        ...


class _SummarizerAdapter:
    """Adapts an async callable into the sync ``Summarizer`` protocol."""

    def __init__(self, fn: Callable[[tuple[Message, ...]], Awaitable[str]]) -> None:
        self._fn = fn

    def summarize(self, messages: tuple[Message, ...]) -> str:
        import anyio

        return anyio.run(self._fn, messages)


@dataclass(slots=True)
class TokenWindowedMemory:
    """Short-term memory bounded by an estimated token budget.

    Keeps all system messages plus the most recent non-system messages whose
    tokens fit within ``max_tokens``. More faithful to a model's context
    window than count-based windowing.
    """

    tokenizer: "object"
    max_tokens: int
    _system_messages: list[Message] = field(default_factory=list)
    _recent: deque[Message] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.max_tokens = max(1, self.max_tokens)

    def add(self, message: Message) -> None:
        if message.role == Role.SYSTEM:
            self._system_messages.append(message)
            return
        self._recent.append(message)
        while len(self._recent) > 1 and self._total_tokens() > self.max_tokens:
            self._recent.popleft()

    def history(self) -> tuple[Message, ...]:
        return (*self._system_messages, *self._recent)

    def _total_tokens(self) -> int:
        total = 0
        for m in self._system_messages:
            total += self.tokenizer.count_tokens(m.content)  # type: ignore[attr-defined]
        for m in self._recent:
            total += self.tokenizer.count_tokens(m.content)  # type: ignore[attr-defined]
        return total


_SUMMARY_PREFIX = "Summary of earlier conversation:\n"


@dataclass(slots=True)
class SummarizingMemory:
    """Memory that stays within a token budget by rolling older turns into a summary.

    System messages and the most recent ``min_recent`` non-system messages are
    kept verbatim; once the budget is exceeded, the oldest non-system messages
    are folded into a running summary via the ``Summarizer``.
    """

    tokenizer: "object"
    summarizer: Summarizer
    max_tokens: int
    min_recent: int = 4
    _system_messages: list[Message] = field(default_factory=list)
    _recent: deque[Message] = field(default_factory=deque)
    _summary: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.max_tokens = max(1, self.max_tokens)
        self.min_recent = max(1, self.min_recent)

    def add(self, message: Message) -> None:
        if message.role == Role.SYSTEM:
            self._system_messages.append(message)
            return
        self._recent.append(message)
        while len(self._recent) > self.min_recent and self._total_tokens() > self.max_tokens:
            self._fold(self._recent.popleft())

    def history(self) -> tuple[Message, ...]:
        result: list[Message] = list(self._system_messages)
        if self._summary is not None:
            result.append(Message.system(_SUMMARY_PREFIX + self._summary))
        result.extend(self._recent)
        return tuple(result)

    def _total_tokens(self) -> int:
        total = 0
        for m in self._system_messages:
            total += self.tokenizer.count_tokens(m.content)  # type: ignore[attr-defined]
        if self._summary is not None:
            total += self.tokenizer.count_tokens(self._summary)  # type: ignore[attr-defined]
        for m in self._recent:
            total += self.tokenizer.count_tokens(m.content)  # type: ignore[attr-defined]
        return total

    def _fold(self, message: Message) -> None:
        if self._summary is not None:
            combined: tuple[Message, ...] = (Message.system(self._summary), message)
        else:
            combined = (message,)
        self._summary = self.summarizer.summarize(combined)
