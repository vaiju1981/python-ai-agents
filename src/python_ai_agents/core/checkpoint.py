from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import anyio


@dataclass(frozen=True, slots=True)
class Checkpoint:
    tenant: str
    run_id: str
    payload_json: str


class CheckpointStore(Protocol):
    async def load(self, tenant: str, run_id: str) -> Checkpoint | None:
        ...

    async def save(self, checkpoint: Checkpoint) -> None:
        ...

    async def delete(self, tenant: str, run_id: str) -> None:
        ...


class SQLiteCheckpointStore:
    """SQLite checkpoint store keyed by tenant and run id."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    async def load(self, tenant: str, run_id: str) -> Checkpoint | None:
        return await anyio.to_thread.run_sync(self._load, tenant, run_id)

    async def save(self, checkpoint: Checkpoint) -> None:
        await anyio.to_thread.run_sync(self._save, checkpoint)

    async def delete(self, tenant: str, run_id: str) -> None:
        await anyio.to_thread.run_sync(self._delete, tenant, run_id)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    tenant TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant, run_id)
                )
                """
            )

    def _load(self, tenant: str, run_id: str) -> Checkpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT tenant, run_id, payload_json
                FROM checkpoints
                WHERE tenant = ? AND run_id = ?
                """,
                (tenant, run_id),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(tenant=row[0], run_id=row[1], payload_json=row[2])

    def _save(self, checkpoint: Checkpoint) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (tenant, run_id, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant, run_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    checkpoint.tenant,
                    checkpoint.run_id,
                    checkpoint.payload_json,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _delete(self, tenant: str, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM checkpoints WHERE tenant = ? AND run_id = ?",
                (tenant, run_id),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
