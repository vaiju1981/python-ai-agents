"""Durable SQLite-backed audit store for the analytics agent.

The core runtime already records rich, contextual audit events (tool
start/end/timeout/denial, errors) to whatever ``AuditSink`` the agent is given.
This module adds a persistent ``SqliteAuditStore`` that:

- satisfies the core ``AuditSink`` protocol, so governance events are written
  to a durable ``audit_events`` table with tenant/session/trace context;
- also satisfies the ``AgentObserver`` protocol, so per-tool telemetry
  (success, latency, row count, error) is captured into a queryable
  ``tool_calls`` table for the UI;
- exposes ``records(...)`` so the Audit tab can read the durable log back,
  surviving restarts and scaling past an in-memory list.

Use one instance per dataset/session and hand it to ``create_agent`` as both
``audit_sink`` and an ``observer``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio

from python_ai_agents.core.audit import AuditEvent
from python_ai_agents.core.observe import NoopAgentObserver
from python_ai_agents.core.tool import ToolResult


class SqliteAuditStore(NoopAgentObserver):
    """SQLite-backed audit store: durable governance + tool-call telemetry.

    Writes are connection-per-operation (mirroring the core ``SQLiteAuditSink``)
    with a generous lock timeout so low-volume concurrent writes from the agent
    loop and observer hooks do not contend.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Set by the host before each run so observer telemetry (which carries
        # no context of its own) can be grouped under the active session.
        self.session_id: str = ""
        self._initialize()

    # --- AuditSink protocol: governance events from the core runtime ---
    async def record(self, event: AuditEvent) -> None:
        await anyio.to_thread.run_sync(self._record_event, event)

    # --- AgentObserver protocol: rich per-tool telemetry for the UI ---
    async def on_tool_result(
        self, tool_name: str, result: ToolResult, latency: object
    ) -> None:
        rows = result.data
        n_rows = len(rows) if isinstance(rows, list) else None
        await anyio.to_thread.run_sync(
            self._record_call,
            tool_name,
            not result.error,
            _seconds(latency),
            n_rows,
            (result.content[:200] if result.error else ""),
        )

    async def on_error(self, stage: str, error: BaseException) -> None:
        await anyio.to_thread.run_sync(
            self._record_call,
            f"<{stage}>",
            False,
            None,
            None,
            f"{error.__class__.__name__}: {error}",
        )

    # --- Read API for the UI ---
    def records(
        self,
        *,
        session_id: str | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        """Return tool-call audit rows (newest first) for the Audit tab."""
        where: list[str] = []
        values: list[object] = []
        if session_id:
            where.append("session_id = ?")
            values.append(session_id)

        query = (
            "SELECT id, timestamp, session_id, tool, ok, latency_s, rows, error "
            "FROM tool_calls"
        )
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY timestamp DESC, rowid DESC"
        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [
            {
                "tool": row[3],
                "ok": bool(row[4]),
                "latency_s": round(row[5], 3) if row[5] is not None else None,
                "rows": row[6],
                "error": row[7],
            }
            for row in rows
        ]

    def event_log(
        self,
        *,
        session_id: str | None = None,
        limit: int | None = 200,
    ) -> list[AuditEvent]:
        """Return the core governance audit events (denials, timeouts, errors)."""
        where: list[str] = []
        values: list[object] = []
        if session_id:
            where.append("session_id = ?")
            values.append(session_id)
        query = (
            "SELECT id, timestamp, event_type, trace_id, session_id, principal, "
            "tenant, detail FROM audit_events"
        )
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY timestamp DESC, rowid DESC"
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

    def close(self) -> None:
        return None

    # --- internals ---
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
                "CREATE INDEX IF NOT EXISTS idx_audit_events_session "
                "ON audit_events(session_id)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    trace_id TEXT,
                    event_type TEXT,
                    tool TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    latency_s REAL,
                    rows INTEGER,
                    error TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(timestamp)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def _record_event(self, event: AuditEvent) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO audit_events (
                        id, timestamp, event_type, trace_id, session_id, principal,
                        tenant, detail
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
        except Exception:
            return None

    def _record_call(
        self,
        tool: str,
        ok: bool,
        latency_s: float | None,
        rows: int | None,
        error: str,
    ) -> None:
        try:
            sid = self.session_id or "unknown"
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO tool_calls (
                        id, timestamp, session_id, trace_id, event_type, tool, ok,
                        latency_s, rows, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        datetime.now(timezone.utc).isoformat(),
                        sid,
                        sid,
                        "tool.result",
                        tool,
                        int(ok),
                        latency_s,
                        rows,
                        error or "",
                    ),
                )
        except Exception:
            return None


def _seconds(latency: object) -> float | None:
    if latency is None:
        return None
    total = getattr(latency, "total_seconds", None)
    if callable(total):
        return float(total())
    try:
        return float(latency)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
