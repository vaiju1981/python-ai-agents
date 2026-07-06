"""Tests for model-port wrappers: resilient, replay, observing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import anyio
import pytest

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    DefaultAgent,
    ModelRequest,
    ModelResponse,
    ObservingModelPort,
    RecordingObserver,
    ReplayModelPort,
    ResilientModelPort,
    Usage,
)


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse] | None = None, fail_times: int = 0) -> None:
        self.responses = list(responses or [])
        self.fail_times = fail_times
        self.calls = 0

    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("transient failure")
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse.text_response("ok")


def test_resilient_model_port_retries_on_transient_failure() -> None:
    async def run() -> None:
        model = ScriptedModel(fail_times=2)
        resilient = ResilientModelPort(
            delegate=model, max_attempts=3, backoff_ms=0
        )
        response = await resilient.chat(ModelRequest(messages=()))
        assert response.text == "ok"
        assert model.calls == 3

    anyio.run(run)


def test_resilient_model_port_fails_fast_on_non_retryable() -> None:
    async def run() -> None:
        model = ScriptedModel(fail_times=1)
        resilient = ResilientModelPort(
            delegate=model, max_attempts=3, backoff_ms=0,
            retryable=lambda e: not isinstance(e, ValueError),
        )

        class NonRetryableModel:
            async def chat(self, request: ModelRequest) -> ModelResponse:
                raise ValueError("auth error")

        resilient2 = ResilientModelPort(
            delegate=NonRetryableModel(), max_attempts=3, backoff_ms=0,
            retryable=lambda e: not isinstance(e, ValueError),
        )
        with pytest.raises(ValueError):
            await resilient2.chat(ModelRequest(messages=()))

    anyio.run(run)


def test_replay_model_port_replays_in_order() -> None:
    async def run() -> None:
        responses = [
            ModelResponse.text_response("first"),
            ModelResponse.text_response("second"),
        ]
        replay = ReplayModelPort(recorded=responses)

        r1 = await replay.chat(ModelRequest(messages=()))
        r2 = await replay.chat(ModelRequest(messages=()))
        assert r1.text == "first"
        assert r2.text == "second"

    anyio.run(run)


def test_replay_model_port_exhausts() -> None:
    async def run() -> None:
        replay = ReplayModelPort(recorded=[ModelResponse.text_response("only")])
        await replay.chat(ModelRequest(messages=()))
        with pytest.raises(RuntimeError, match="replay exhausted"):
            await replay.chat(ModelRequest(messages=()))

    anyio.run(run)


def test_replay_model_port_with_default_agent() -> None:
    async def run() -> None:
        replay = ReplayModelPort(recorded=[ModelResponse.text_response("replayed")])
        agent = DefaultAgent(replay)
        response = await agent.run(AgentRequest.ephemeral("hi"))
        assert response.output == "replayed"

    anyio.run(run)


def test_observing_model_port_emits_events() -> None:
    async def run() -> None:
        observer = RecordingObserver()
        model = ObservingModelPort(
            delegate=ScriptedModel(responses=[ModelResponse.text_response("observed")]),
            observers=[observer],
        )
        response = await model.chat(ModelRequest(messages=()))
        assert response.text == "observed"
        assert len(observer.model_requests) == 1
        assert len(observer.model_responses) == 1

    anyio.run(run)


def test_observing_model_port_isolates_observer_failures() -> None:
    async def run() -> None:
        class FailingObserver:
            async def on_model_call(self, request: ModelRequest) -> None:
                raise RuntimeError("observer failed")
            async def on_model_response(self, response: ModelResponse, latency: timedelta) -> None:
                raise RuntimeError("observer failed")

        model = ObservingModelPort(
            delegate=ScriptedModel(responses=[ModelResponse.text_response("ok")]),
            observers=[FailingObserver()],
        )
        response = await model.chat(ModelRequest(messages=()))
        assert response.text == "ok"

    anyio.run(run)
