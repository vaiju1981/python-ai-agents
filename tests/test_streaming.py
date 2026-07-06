"""Tests for the streaming model seam."""

from __future__ import annotations

from collections.abc import AsyncIterator

import anyio

from python_ai_agents import (
    AgentRequest,
    DefaultAgent,
    ModelChunk,
    ModelRequest,
    StreamingModelAdapter,
    ToolCall,
    Usage,
)


class ScriptedStreamingModel:
    """Streams pre-defined chunks for testing."""

    def __init__(self, chunks: list[ModelChunk]) -> None:
        self.chunks = list(chunks)
        self.requests: list[ModelRequest] = []

    async def stream_chat(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        self.requests.append(request)
        for chunk in self.chunks:
            yield chunk


def test_streaming_adapter_accumulates_text_chunks() -> None:
    async def run() -> None:
        model = ScriptedStreamingModel([
            ModelChunk(text="Hello "),
            ModelChunk(text="world"),
            ModelChunk(text="!", done=True),
        ])
        adapter = StreamingModelAdapter(model)

        response = await adapter.chat(ModelRequest(messages=()))

        assert response.text == "Hello world!"
        assert not response.tool_calls

    anyio.run(run)


def test_streaming_adapter_accumulates_tool_calls() -> None:
    async def run() -> None:
        call1 = ToolCall(name="search", arguments={"q": "test"}, id="c1")
        call2 = ToolCall(name="fetch", arguments={"url": "http://x"}, id="c2")
        model = ScriptedStreamingModel([
            ModelChunk(tool_calls=(call1,)),
            ModelChunk(tool_calls=(call2,)),
            ModelChunk(done=True),
        ])
        adapter = StreamingModelAdapter(model)

        response = await adapter.chat(ModelRequest(messages=()))

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "search"
        assert response.tool_calls[1].name == "fetch"

    anyio.run(run)


def test_streaming_adapter_captures_usage() -> None:
    async def run() -> None:
        model = ScriptedStreamingModel([
            ModelChunk(text="partial"),
            ModelChunk(usage=Usage(input_tokens=15, output_tokens=25), done=True),
        ])
        adapter = StreamingModelAdapter(model)

        response = await adapter.chat(ModelRequest(messages=()))

        assert response.usage.input_tokens == 15
        assert response.usage.output_tokens == 25

    anyio.run(run)


def test_streaming_adapter_works_with_default_agent() -> None:
    async def run() -> None:
        model = ScriptedStreamingModel([
            ModelChunk(text="Hello "),
            ModelChunk(text="from stream", done=True),
        ])
        adapter = StreamingModelAdapter(model)
        agent = DefaultAgent(adapter)

        response = await agent.run(AgentRequest.ephemeral("hi"))

        assert response.is_completed
        assert response.output == "Hello from stream"

    anyio.run(run)


def test_streaming_adapter_handles_empty_stream() -> None:
    async def run() -> None:
        model = ScriptedStreamingModel([])
        adapter = StreamingModelAdapter(model)

        response = await adapter.chat(ModelRequest(messages=()))

        assert response.text == ""
        assert response.tool_calls == ()

    anyio.run(run)


def test_streaming_adapter_last_usage_wins() -> None:
    async def run() -> None:
        model = ScriptedStreamingModel([
            ModelChunk(text="a", usage=Usage(input_tokens=5, output_tokens=5)),
            ModelChunk(text="b", usage=Usage(input_tokens=10, output_tokens=20), done=True),
        ])
        adapter = StreamingModelAdapter(model)

        response = await adapter.chat(ModelRequest(messages=()))

        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 20

    anyio.run(run)
