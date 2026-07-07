"""Model-port wrappers: resilient retry, replay, and observing.

* ``ResilientModelPort`` — per-call timeout + jittered retry backoff.
* ``ReplayModelPort`` — replays recorded responses for deterministic replay.
* ``ObservingModelPort`` — emits observer events for model calls outside
  ``DefaultAgent`` (e.g. planners, summarizers).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import anyio

from python_ai_agents.core.model import ModelPort, ModelRequest, ModelResponse
from python_ai_agents.core.observe import AgentObserver

__all__ = [
    "ObservingModelPort",
    "ReplayModelPort",
    "ResilientModelPort",
]


@dataclass(slots=True)
class ResilientModelPort:
    """Wraps a ``ModelPort`` with per-call timeout and bounded retries.

    Each ``chat`` call is bounded by ``timeout_seconds``. On failure the call
    is retried up to ``max_attempts`` times with jittered linear backoff
    (full-jitter strategy to avoid thundering herds). Exceptions matching
    ``retryable`` are retried; others fail fast.
    """

    delegate: ModelPort
    max_attempts: int = 3
    timeout_seconds: float = 60.0
    backoff_ms: float = 500.0
    retryable: Callable[[BaseException], bool] = field(default=lambda _e: True)

    async def chat(self, request: ModelRequest) -> ModelResponse:
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result: ModelResponse | None = None
                with anyio.move_on_after(self.timeout_seconds) as scope:
                    result = await self.delegate.chat(request)
                if scope.cancel_called:
                    raise TimeoutError(
                        f"model call timed out after {self.timeout_seconds}s"
                    )
                return result
            except Exception as exc:
                if not self.retryable(exc) or attempt >= self.max_attempts:
                    raise
                last_exc = exc
                delay = self._jittered_backoff(attempt)
                if delay > 0:
                    await anyio.sleep(delay / 1000.0)
        # Should not reach here, but satisfy the type checker.
        raise last_exc  # type: ignore[misc]

    def _jittered_backoff(self, attempt: int) -> float:
        ceiling = self.backoff_ms * attempt
        if ceiling <= 0:
            return 0.0
        return random.uniform(0, ceiling)


@dataclass(slots=True)
class ReplayModelPort:
    """Replays recorded ``ModelResponse`` values in order for deterministic replay.

    Combined with ``RecordingObserver`` and a replay tool executor, this
    reproduces a run with fixed model outputs and no repeated side effects.
    """

    recorded: list[ModelResponse]
    _index: int = field(default=0, init=False)

    async def chat(self, request: ModelRequest) -> ModelResponse:
        if self._index >= len(self.recorded):
            raise RuntimeError(
                f"replay exhausted: no recorded response for call #{self._index}"
            )
        response = self.recorded[self._index]
        self._index += 1
        return response


@dataclass(slots=True)
class ObservingModelPort:
    """Decorates a ``ModelPort`` so each call emits observer events.

    Useful for model calls outside ``DefaultAgent`` (e.g. a planner or
    summarizer). Observer failures are isolated.
    """

    delegate: ModelPort
    observers: list[AgentObserver] = field(default_factory=list)

    async def chat(self, request: ModelRequest) -> ModelResponse:
        for observer in self.observers:
            try:
                await observer.on_model_call(request)
            except Exception:
                pass
        response = await self.delegate.chat(request)
        for observer in self.observers:
            try:
                await observer.on_model_response(response, _zero_duration)
            except Exception:
                pass
        return response


_zero_duration = timedelta(0)
