"""Replay tool executor for deterministic replay of recorded runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from python_ai_agents.core.tool import ToolResult

__all__ = ["ReplayToolExecutor"]


@dataclass
class ReplayToolExecutor:
    """Returns recorded tool results without re-running the real tools.

    Keyed by ``(tool_name, arguments_json)``. Combined with ``ReplayModelPort``
    and ``RecordingObserver``, this gives deterministic replay with no
    repeated side effects.
    """

    recorded: dict[tuple[str, str], ToolResult] = field(default_factory=dict)
    _index: int = field(default=0, init=False)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        key = (tool_name, json.dumps(arguments, sort_keys=True))
        if key in self.recorded:
            return self.recorded[key]
        # Fallback: return results in insertion order for non-keyed replay
        results = list(self.recorded.values())
        if self._index < len(results):
            result = results[self._index]
            self._index += 1
            return result
        return ToolResult.failed("replay exhausted: no recorded result")
