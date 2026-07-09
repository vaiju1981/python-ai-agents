"""Decision / approval governance store (generic lift of ATLAS decisions.py).

A JSON-backed, append-only audit log for high-impact recommendations. Each
recommendation has a verb status (accepted/rejected/deferred/pending), a governed
lifecycle stage (pending → approved → scheduled → implemented → ...), and a
host-notify state. Approved-but-not-yet-implemented actions are exposed so an
optimizer can reserve committed capital.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demos.analytics.src.analytics.file_lock import atomic_write_text, file_lock

STATUSES = {"accepted", "rejected", "deferred", "pending"}
STAGES = ["pending", "approved", "scheduled", "implemented", "rejected", "deferred"]
VERB_TO_STAGE = {
    "accepted": "approved",
    "rejected": "rejected",
    "deferred": "deferred",
}
NOTIFY_STATES = {"not_requested", "requested", "acknowledged"}


@dataclass
class DecisionEntry:
    rid: str
    unit_id: str
    action_type: str
    status: str
    stage: str
    comment: str
    by: str
    history: list[dict[str, Any]] = field(default_factory=list)
    scheduled_for: str | None = None
    notify_state: str = "not_requested"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rid": self.rid,
            "unitId": self.unit_id,
            "actionType": self.action_type,
            "status": self.status,
            "stage": self.stage,
            "comment": self.comment,
            "by": self.by,
            "scheduledFor": self.scheduled_for,
            "notifyState": self.notify_state,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecisionEntry:
        return cls(
            rid=d["rid"],
            unit_id=d.get("unitId", d.get("unit_id", "")),
            action_type=d.get("actionType", d.get("action_type", "")),
            status=d.get("status", "pending"),
            stage=d.get("stage", "pending"),
            comment=d.get("comment", ""),
            by=d.get("by", "operator"),
            history=d.get("history", []),
            scheduled_for=d.get("scheduledFor", d.get("scheduled_for")),
            notify_state=d.get("notifyState", d.get("notify_state", "not_requested")),
        )


class DecisionStore:
    """Thread-safe JSON store of decision records."""

    def __init__(self, file: Path | str | None = None) -> None:
        self.file = Path(file) if file else None
        self._lock = threading.Lock()
        self._entries: dict[str, DecisionEntry] = {}
        self._outcomes: list[dict[str, Any]] = []
        if file and Path(file).exists():
            try:
                with file_lock(Path(file)):
                    raw = json.loads(Path(file).read_text())
                for e in raw.get("entries", []):
                    self._entries[e["rid"]] = DecisionEntry.from_dict(e)
                self._outcomes = list(raw.get("outcomes", []))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    def _rid(self, unit_id: str, action_type: str) -> str:
        return f"{unit_id}::{action_type}"

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Hold both the in-process and cross-process lock, and make the
        in-memory state reflect the latest on-disk contents before mutating.

        Wrapping the whole read-modify-write (not just the save) is what makes
        concurrent worker *processes* safe: each mutation reloads the current
        file, applies its change, then atomically writes back.
        """
        with self._lock:
            with file_lock(self.file):
                self._reload()
                try:
                    yield
                finally:
                    self._save()

    def _reload(self) -> None:
        if self.file is None or not Path(self.file).exists():
            return
        try:
            raw = json.loads(Path(self.file).read_text())
        except (json.JSONDecodeError, KeyError, TypeError):
            return
        self._entries = {e["rid"]: DecisionEntry.from_dict(e) for e in raw.get("entries", [])}
        self._outcomes = list(raw.get("outcomes", []))

    def record(
        self,
        unit_id: str,
        action_type: str,
        status: str,
        comment: str = "",
        by: str = "operator",
        now: datetime | None = None,
    ) -> DecisionEntry:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        now = now or datetime.now(timezone.utc)
        rid = self._rid(unit_id, action_type)
        with self._exclusive():
            existing = self._entries.get(rid)
            stage = VERB_TO_STAGE.get(status, "pending")
            entry = existing or DecisionEntry(
                rid=rid,
                unit_id=unit_id,
                action_type=action_type,
                status=status,
                stage=stage,
                comment=comment,
                by=by,
            )
            if existing:
                # Never regress an approved/scheduled/implemented approval.
                if existing.stage in ("scheduled", "implemented") and stage in (
                    "approved",
                    "pending",
                ):
                    stage = existing.stage
                entry.status = status
                entry.stage = stage
                entry.comment = comment
                entry.by = by
            entry.history.append(
                {
                    "at": now.isoformat(),
                    "status": status,
                    "stage": entry.stage,
                    "comment": comment,
                    "by": by,
                }
            )
            self._entries[rid] = entry
            return entry

    def set_stage(
        self,
        unit_id: str,
        action_type: str,
        stage: str,
        scheduled_for: str | None = None,
        by: str = "operator",
        comment: str = "",
        now: datetime | None = None,
    ) -> DecisionEntry:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        now = now or datetime.now(timezone.utc)
        rid = self._rid(unit_id, action_type)
        with self._exclusive():
            entry = self._entries.get(rid)
            if entry is None:
                entry = DecisionEntry(
                    rid=rid,
                    unit_id=unit_id,
                    action_type=action_type,
                    status="pending",
                    stage=stage,
                    comment=comment,
                    by=by,
                )
            else:
                entry.stage = stage
                entry.by = by
                if comment:
                    entry.comment = comment
            if scheduled_for:
                entry.scheduled_for = scheduled_for
            entry.history.append(
                {"at": now.isoformat(), "stage": stage, "comment": comment, "by": by}
            )
            self._entries[rid] = entry
            return entry

    def notify_host(
        self,
        unit_id: str,
        action_type: str,
        note: str = "",
        by: str = "system",
        now: datetime | None = None,
    ) -> DecisionEntry:
        now = now or datetime.now(timezone.utc)
        rid = self._rid(unit_id, action_type)
        with self._exclusive():
            entry = self._entries.get(rid)
            if entry is None:
                entry = DecisionEntry(
                    rid=rid,
                    unit_id=unit_id,
                    action_type=action_type,
                    status="pending",
                    stage="pending",
                    comment=note,
                    by=by,
                )
            entry.notify_state = "requested"
            entry.history.append(
                {"at": now.isoformat(), "notify": "requested", "note": note, "by": by}
            )
            self._entries[rid] = entry
            return entry

    def approved_actions(self) -> list[str]:
        """Action types currently approved (reserved capital for the optimizer)."""
        return [
            e.action_type for e in self._entries.values() if e.stage in ("approved", "scheduled")
        ]

    def board(self) -> dict[str, Any]:
        counts: dict[str, int] = {s: 0 for s in STAGES}
        for e in self._entries.values():
            counts[e.stage] = counts.get(e.stage, 0) + 1
        return {
            "stages": counts,
            "counts": len(self._entries),
            "notifyRequested": sum(
                1 for e in self._entries.values() if e.notify_state == "requested"
            ),
            "total": len(self._entries),
        }

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_stage: dict[str, int] = {}
        for e in self._entries.values():
            by_status[e.status] = by_status.get(e.status, 0) + 1
            by_stage[e.stage] = by_stage.get(e.stage, 0) + 1
        return {
            "total": len(self._entries),
            "byStatus": by_status,
            "byStage": by_stage,
            "nWithComment": sum(1 for e in self._entries.values() if e.comment),
            "notifyRequested": sum(
                1 for e in self._entries.values() if e.notify_state == "requested"
            ),
            "recent": [h for e in self._entries.values() for h in e.history[-1:]][:10],
        }

    def all(self) -> list[DecisionEntry]:
        return list(self._entries.values())

    # ------------------------------------------------------------------
    # Feedback loop: capture recommendation outcomes and tune trust bars.
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        action_type: str,
        decision: str,
        trust_tier: str | None = None,
        correct: bool | None = None,
        unit_id: str = "",
        comment: str = "",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Record the realized outcome of a recommendation for later tuning."""
        now = now or datetime.now(timezone.utc)
        outcome = {
            "at": now.isoformat(),
            "actionType": action_type,
            "unitId": unit_id,
            "decision": decision,
            "trustTier": trust_tier,
            "correct": correct,
            "comment": comment,
        }
        with self._exclusive():
            self._outcomes.append(outcome)
        return outcome

    def outcomes(self) -> list[dict[str, Any]]:
        return list(self._outcomes)

    def tune_trust_thresholds(self, min_samples: int = 3) -> dict[str, Any]:
        """Suggest raising/lowering the trust bar from labeled outcomes.

        If TRUSTED recommendations are frequently wrong, recommend *raising* the
        bar; if lower tiers are frequently right, recommend *lowering* it.
        """
        by_tier: dict[str, list[bool]] = {}
        for o in self._outcomes:
            tier = o.get("trustTier")
            correct = o.get("correct")
            if tier is None or correct is None:
                continue
            by_tier.setdefault(tier, []).append(bool(correct))

        trusted = by_tier.get("TRUSTED", [])
        action = "hold"
        reason = "insufficient labeled outcomes to tune"
        accuracy = None
        if len(trusted) >= min_samples:
            accuracy = sum(trusted) / len(trusted)
            if accuracy < 0.5:
                action = "raise"
                reason = f"TRUSTED accuracy {accuracy:.0%} below 50%; raise the bar"
            elif accuracy >= 0.9:
                action = "hold"
                reason = f"TRUSTED accuracy {accuracy:.0%} healthy"
        # Consider promoting a lower tier that performs well.
        directional = by_tier.get("DIRECTIONAL", [])
        if action == "hold" and len(directional) >= min_samples:
            d_acc = sum(directional) / len(directional)
            if d_acc >= 0.9:
                action = "lower"
                reason = f"DIRECTIONAL accuracy {d_acc:.0%}; safe to lower the bar"
        return {
            "action": action,
            "reason": reason,
            "trustedAccuracy": accuracy,
            "samplesByTier": {k: len(v) for k, v in by_tier.items()},
        }

    def close(self) -> None:
        """Flush pending state to disk (state is already persisted per-op)."""
        with self._exclusive():
            pass

    def _save(self) -> None:
        # Callers must already hold both ``self._lock`` and the cross-process
        # ``file_lock`` (see ``_exclusive``); do NOT re-acquire here, since
        # ``fcntl.flock`` is not recursive and would self-deadlock.
        if self.file is None:
            return
        data = {
            "entries": [e.to_dict() for e in self._entries.values()],
            "outcomes": self._outcomes,
        }
        atomic_write_text(self.file, json.dumps(data, indent=2))
