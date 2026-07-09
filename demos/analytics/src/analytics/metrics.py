"""Structured metrics for the analytics engine (PR-4 observability).

The engine is a long-running service in production, so trust grades, drift,
latency, and error rates must be observable — not buried inside answer
payloads. This module provides a tiny metrics facade with a pluggable sink:

- ``LogMetricsSink`` (default): emits one JSON line per metric via ``logging``.
- ``InMemoryMetricsSink``: for tests / in-process aggregation.
- a real Prometheus / OTLP sink can be dropped in later behind the same
  ``MetricsSink`` protocol with **no change to call sites** (no new hard
  dependency).

Call sites use the module helpers ``inc`` / ``set_gauge`` / ``observe``. The
answer path (``toolset._ok`` / ``_make_tool``), drift detection, and the model
cache all emit through here.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Protocol


class MetricsSink(Protocol):
    def emit(self, name: str, kind: str, value: float, tags: dict[str, str]) -> None:
        """Record a metric ``name`` of ``kind`` (counter/gauge/histogram)."""


class LogMetricsSink:
    """Default sink: one JSON line per metric via the ``analytics.metrics`` logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or logging.getLogger("analytics.metrics")

    def emit(self, name: str, kind: str, value: float, tags: dict[str, str]) -> None:
        self._log.info(json.dumps({"metric": name, "type": kind, "value": value, "tags": tags}))


class InMemoryMetricsSink:
    """Sink that records events in memory — for tests and in-process rollups."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, name: str, kind: str, value: float, tags: dict[str, str]) -> None:
        self.events.append({"metric": name, "type": kind, "value": value, "tags": dict(tags)})

    def _match(self, name: str, tags: dict[str, str]) -> list[dict[str, Any]]:
        return [
            e
            for e in self.events
            if e["metric"] == name
            and all(e["tags"].get(k) == v for k, v in tags.items())
        ]

    def count(self, name: str, **tags: str) -> float:
        """Sum of all ``counter`` values for ``name`` (default 0.0)."""
        return float(sum(e["value"] for e in self._match(name, tags)))

    def values(self, name: str, **tags: str) -> list[float]:
        """All recorded values for ``name`` (any kind)."""
        return [e["value"] for e in self._match(name, tags)]

    def tags_seen(self, name: str) -> set[tuple[str, str]]:
        """Distinct tag-combos observed for ``name`` (handy in tests)."""
        return {(k, v) for e in self._match(name, {}) for k, v in e["tags"].items()}


# The tool currently executing, so the answer path can tag metrics with the
# tool name without threading it through every ``_ok`` call site.
_current_tool: ContextVar[str | None] = ContextVar("analytics_current_tool", default=None)

_default_sink: MetricsSink = LogMetricsSink()


def set_sink(sink: MetricsSink) -> None:
    """Install a process-wide metrics sink (e.g. Prometheus/OTLP in prod)."""
    global _default_sink
    _default_sink = sink


def get_sink() -> MetricsSink:
    return _default_sink


def set_current_tool(name: str | None) -> None:
    _current_tool.set(name)


def current_tool() -> str | None:
    return _current_tool.get()


def _emit(name: str, kind: str, value: float, tags: dict[str, str] | None) -> None:
    _default_sink.emit(name, kind, float(value), tags or {})


def inc(name: str, tags: dict[str, str] | None = None, value: float = 1.0) -> None:
    _emit(name, "counter", value, tags)


def set_gauge(name: str, value: float, tags: dict[str, str] | None = None) -> None:
    _emit(name, "gauge", value, tags)


def observe(name: str, value: float, tags: dict[str, str] | None = None) -> None:
    _emit(name, "histogram", value, tags)


def readiness(directories: list[str] | None = None) -> dict[str, Any]:
    """Liveness/readiness probe: reports the active sink and store connectivity.

    ``directories`` should be the model/decision/audit store roots the process
    relies on (e.g. the ``FileModelStore`` directory). When omitted, the env var
    ``ANALYTICS_MODEL_CACHE_DIR`` is used if set. Each directory is checked for
    existence, readability, and writability (a tiny probe file is created + removed).
    """
    import os

    dirs = list(directories or [])
    env_dir = os.getenv("ANALYTICS_MODEL_CACHE_DIR")
    if not dirs and env_dir:
        dirs = [env_dir]

    stores: dict[str, Any] = {}
    all_ok = True
    for d in dirs:
        d = str(d)
        info: dict[str, Any] = {"exists": False, "readable": False, "writable": False}
        try:
            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            info["exists"] = p.is_dir()
            probe = p / ".readiness_probe"
            probe.write_text("ok")
            info["writable"] = True
            _ = probe.read_text()
            info["readable"] = True
            probe.unlink()
        except Exception as exc:  # pragma: no cover - environment-dependent
            info["error"] = str(exc)
            all_ok = False
        if not (info["exists"] and info["readable"] and info["writable"]):
            all_ok = False
        stores[d] = info

    return {
        "ok": all_ok,
        "sink": type(_default_sink).__name__,
        "stores": stores,
    }
