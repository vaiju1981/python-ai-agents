from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

from python_ai_agents.core.agent import AgentResponse
from python_ai_agents.core.model import Message, ModelRequest, ModelResponse, ToolCall, Usage
from python_ai_agents.core.tool import ToolResult


class AgentObserver(Protocol):
    async def on_turn_start(self, input_text: str) -> None: ...

    async def on_model_call(self, request: ModelRequest) -> None: ...

    async def on_model_response(self, response: ModelResponse, latency: timedelta) -> None: ...

    async def on_usage(self, model: str, usage: Usage) -> None: ...

    async def on_tool_call(self, call: ToolCall) -> None: ...

    async def on_tool_result(
        self, tool_name: str, result: ToolResult, latency: timedelta
    ) -> None: ...

    async def on_turn_end(self, response: AgentResponse, duration: timedelta) -> None: ...

    async def on_error(self, stage: str, error: BaseException) -> None: ...


class NoopAgentObserver:
    async def on_turn_start(self, input_text: str) -> None:
        return None

    async def on_model_call(self, request: ModelRequest) -> None:
        return None

    async def on_model_response(self, response: ModelResponse, latency: timedelta) -> None:
        return None

    async def on_usage(self, model: str, usage: Usage) -> None:
        return None

    async def on_tool_call(self, call: ToolCall) -> None:
        return None

    async def on_tool_result(self, tool_name: str, result: ToolResult, latency: timedelta) -> None:
        return None

    async def on_turn_end(self, response: AgentResponse, duration: timedelta) -> None:
        return None

    async def on_error(self, stage: str, error: BaseException) -> None:
        return None


class RecordingObserver(NoopAgentObserver):
    def __init__(self) -> None:
        self.turn_inputs: list[str] = []
        self.model_requests: list[ModelRequest] = []
        self.model_responses: list[ModelResponse] = []
        self.tool_calls: list[ToolCall] = []
        self.tool_results: list[ToolResult] = []
        self.turn_responses: list[AgentResponse] = []
        self.errors: list[tuple[str, str]] = []

    async def on_turn_start(self, input_text: str) -> None:
        self.turn_inputs.append(input_text)

    async def on_model_call(self, request: ModelRequest) -> None:
        self.model_requests.append(request)

    async def on_model_response(self, response: ModelResponse, latency: timedelta) -> None:
        self.model_responses.append(response)

    async def on_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    async def on_tool_result(self, tool_name: str, result: ToolResult, latency: timedelta) -> None:
        self.tool_results.append(result)

    async def on_turn_end(self, response: AgentResponse, duration: timedelta) -> None:
        self.turn_responses.append(response)

    async def on_error(self, stage: str, error: BaseException) -> None:
        self.errors.append((stage, error.__class__.__name__))


@dataclass(frozen=True, slots=True)
class RedactingObserver(NoopAgentObserver):
    delegate: AgentObserver
    replacement: str = "[redacted]"

    async def on_turn_start(self, input_text: str) -> None:
        await self.delegate.on_turn_start(self.replacement)

    async def on_model_call(self, request: ModelRequest) -> None:
        await self.delegate.on_model_call(
            ModelRequest(
                messages=tuple(
                    _redact_message(message, self.replacement) for message in request.messages
                ),
                tools=request.tools,
            )
        )

    async def on_model_response(self, response: ModelResponse, latency: timedelta) -> None:
        await self.delegate.on_model_response(
            ModelResponse(
                text=self.replacement,
                tool_calls=tuple(
                    _redact_call(call, self.replacement) for call in response.tool_calls
                ),
                usage=response.usage,
            ),
            latency,
        )

    async def on_usage(self, model: str, usage: Usage) -> None:
        await self.delegate.on_usage(model, usage)

    async def on_tool_call(self, call: ToolCall) -> None:
        await self.delegate.on_tool_call(_redact_call(call, self.replacement))

    async def on_tool_result(self, tool_name: str, result: ToolResult, latency: timedelta) -> None:
        await self.delegate.on_tool_result(
            tool_name,
            ToolResult(content=self.replacement, error=result.error),
            latency,
        )

    async def on_turn_end(self, response: AgentResponse, duration: timedelta) -> None:
        await self.delegate.on_turn_end(
            AgentResponse(
                output=self.replacement,
                blocked=response.blocked,
                stop_reason=response.stop_reason,
            ),
            duration,
        )

    async def on_error(self, stage: str, error: BaseException) -> None:
        await self.delegate.on_error(stage, RuntimeError(f"redacted {error.__class__.__name__}"))


class TokenAccountingObserver(NoopAgentObserver):
    def __init__(self) -> None:
        self.model_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    async def on_usage(self, model: str, usage: Usage) -> None:
        self.model_calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        if usage.total_tokens is not None:
            self.total_tokens += usage.total_tokens
        else:
            self.total_tokens += (usage.input_tokens or 0) + (usage.output_tokens or 0)


def _redact_message(message: Message, replacement: str) -> Message:
    return Message(
        role=message.role,
        content=replacement,
        tool_calls=tuple(_redact_call(call, replacement) for call in message.tool_calls),
        tool_call_id=message.tool_call_id,
        tool_name=message.tool_name,
    )


def _redact_call(call: ToolCall, replacement: str) -> ToolCall:
    return ToolCall(name=call.name, arguments={"redacted": replacement}, id=call.id)
