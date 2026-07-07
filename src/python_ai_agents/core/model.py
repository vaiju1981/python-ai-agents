from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

from python_ai_agents.core.tool import ToolSpec


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(Role.ASSISTANT, content, tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, tool_name: str, content: str) -> Message:
        return cls(Role.TOOL, content, tool_call_id=tool_call_id, tool_name=tool_name)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)

    @classmethod
    def text_response(cls, text: str, usage: Usage | None = None) -> ModelResponse:
        return cls(text=text, usage=usage or Usage())

    @classmethod
    def tool_response(
        cls,
        tool_calls: tuple[ToolCall, ...],
        text: str = "",
        usage: Usage | None = None,
    ) -> ModelResponse:
        return cls(text=text, tool_calls=tool_calls, usage=usage or Usage())

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ModelPort(Protocol):
    async def chat(self, request: ModelRequest) -> ModelResponse: ...
