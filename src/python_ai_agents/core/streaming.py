"""Streaming model seam.

Extends the ``ModelPort`` protocol with an optional ``stream_chat`` method
that yields incremental ``ModelChunk`` values.  A ``StreamingModelAdapter``
adapts any ``StreamingModelPort`` back to the synchronous ``ModelPort`` seam
by accumulating chunks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from python_ai_agents.core.model import ModelRequest, ModelResponse, ToolCall, Usage

__all__ = [
    "ModelChunk",
    "StreamingModelAdapter",
    "StreamingModelPort",
]


@dataclass(frozen=True, slots=True)
class ModelChunk:
    """Incremental model output chunk."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    done: bool = False


class StreamingModelPort(Protocol):
    """Model seam that streams incremental chunks.

    Implementations should be async generators (``async def`` with ``yield``).
    """

    def stream_chat(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        ...


@dataclass(slots=True)
class StreamingModelAdapter:
    """Adapts a ``StreamingModelPort`` to the ``ModelPort`` protocol.

    Accumulates streamed chunks into a single ``ModelResponse`` so that
    existing agents (``DefaultAgent``, etc.) can consume streaming models
    without modification.
    """

    delegate: StreamingModelPort

    async def chat(self, request: ModelRequest) -> ModelResponse:
        text_parts: list[str] = []
        # Assemble streamed tool calls by id: a provider emits one call across
        # several deltas (same id, argument fragments), so merge by id instead of
        # treating each delta as a separate call. Deltas with no id can't be
        # assembled, so each is kept on its own.
        calls: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        anon = 0
        usage = Usage()

        async for chunk in self.delegate.stream_chat(request):
            if chunk.text:
                text_parts.append(chunk.text)
            for call in chunk.tool_calls:
                if call.id:
                    key = call.id
                else:
                    key, anon = f"__anon_{anon}", anon + 1
                if key in calls:
                    acc = calls[key]
                    if call.name:
                        acc["name"] = call.name
                    acc["arguments"].update(call.arguments or {})
                else:
                    calls[key] = {
                        "name": call.name,
                        "arguments": dict(call.arguments),
                        "id": call.id,
                    }
                    order.append(key)
            if chunk.usage is not None:
                usage = chunk.usage

        tool_calls = tuple(
            ToolCall(name=calls[k]["name"], arguments=calls[k]["arguments"], id=calls[k]["id"])
            for k in order
        )
        return ModelResponse(text="".join(text_parts), tool_calls=tool_calls, usage=usage)
