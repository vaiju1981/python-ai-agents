"""Streaming model seam.

Extends the ``ModelPort`` protocol with an optional ``stream_chat`` method
that yields incremental ``ModelChunk`` values.  A ``StreamingModelAdapter``
adapts any ``StreamingModelPort`` back to the synchronous ``ModelPort`` seam
by accumulating chunks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

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
        tool_calls: list[ToolCall] = []
        usage = Usage()

        async for chunk in self.delegate.stream_chat(request):
            if chunk.text:
                text_parts.append(chunk.text)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.usage is not None:
                usage = chunk.usage

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            usage=usage,
        )
