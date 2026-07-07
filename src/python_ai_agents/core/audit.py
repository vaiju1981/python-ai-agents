from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import anyio

from python_ai_agents.core.context import RequestContext


@dataclass(frozen=True, slots=True)
class AuditEvent:
    id: str
    timestamp: datetime
    event_type: str
    trace_id: str
    session_id: str
    principal: str
    tenant: str
    detail: str

    @classmethod
    def now(cls, event_type: str, context: RequestContext, detail: str = "") -> AuditEvent:
        return cls(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            trace_id=context.trace_id or context.session_id,
            session_id=context.session_id,
            principal=context.principal,
            tenant=context.tenant,
            detail=detail,
        )


class AuditSink(Protocol):
    async def record(self, event: AuditEvent) -> None: ...


class NullAuditSink:
    async def record(self, event: AuditEvent) -> None:
        return None


class InMemoryAuditSink:
    """Keeps audit events in memory for tests and local inspection."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self._events.append(event)

    def events(
        self,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> list[AuditEvent]:
        events = list(self._events)
        if trace_id is not None:
            events = [event for event in events if event.trace_id == trace_id]
        if session_id is not None:
            events = [event for event in events if event.session_id == session_id]
        return events


class SQLiteAuditSink:
    """SQLite-backed audit sink for local product runtimes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    async def record(self, event: AuditEvent) -> None:
        await anyio.to_thread.run_sync(self._record, event)

    def events(
        self,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        where: list[str] = []
        values: list[object] = []
        if trace_id is not None:
            where.append("trace_id = ?")
            values.append(trace_id)
        if session_id is not None:
            where.append("session_id = ?")
            values.append(session_id)

        query = (
            "SELECT id, timestamp, event_type, trace_id, session_id, principal, tenant, detail "
            "FROM audit_events"
        )
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY timestamp ASC, rowid ASC"
        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            AuditEvent(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                event_type=row[2],
                trace_id=row[3],
                session_id=row[4],
                principal=row[5],
                tenant=row[6],
                detail=row[7],
            )
            for row in rows
        ]

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    principal TEXT NOT NULL,
                    tenant TEXT NOT NULL,
                    detail TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_trace ON audit_events(trace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_events_session ON audit_events(session_id)"
            )

    def _record(self, event: AuditEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    id, timestamp, event_type, trace_id, session_id, principal, tenant, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.trace_id,
                    event.session_id,
                    event.principal,
                    event.tenant,
                    event.detail,
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
