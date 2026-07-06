from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anyio

from python_ai_agents.core.agent import AgentRequest, AgentResponse
from python_ai_agents.core.audit import AuditEvent, AuditSink, NullAuditSink
from python_ai_agents.core.model import Message, ModelPort, ModelRequest, ModelResponse
from python_ai_agents.core.tool import (
    AllTools,
    DenyEffectfulTools,
    NoopToolArgumentValidator,
    Tool,
    ToolApprover,
    ToolArgumentValidator,
    ToolResult,
    ToolSelector,
)


MAX_STEPS_MESSAGE = "I couldn't finish that within my step budget. Please try rephrasing."
MODEL_ERROR_MESSAGE = "I ran into a problem reaching the model. Please try again."


@dataclass(slots=True)
class DefaultAgent:
    """Small model/tool loop over the ModelPort seam."""

    model: ModelPort
    tools: list[Tool] = field(default_factory=list)
    system_prompt: str | None = None
    max_steps: int = 8
    tool_selector: ToolSelector = field(default_factory=AllTools)
    argument_validator: ToolArgumentValidator = field(default_factory=NoopToolArgumentValidator)
    tool_approver: ToolApprover = field(default_factory=DenyEffectfulTools)
    audit_sink: AuditSink = field(default_factory=NullAuditSink)
    tool_timeout_seconds: float | None = 30.0
    max_tool_result_chars: int = 8_000
    frame_tool_results: bool = True

    async def run(self, request: AgentRequest) -> AgentResponse:
        history: list[Message] = []
        if self.system_prompt:
            history.append(Message.system(self.system_prompt))
        history.append(Message.user(request.input))

        active_tools = self.tool_selector.select(request.input, list(self.tools), request.context)
        tool_by_name = {tool.spec.name: tool for tool in active_tools}
        tool_specs = tuple(tool.spec for tool in active_tools)

        for _step in range(max(1, self.max_steps)):
            if _deadline_exceeded(request):
                return AgentResponse.stopped("I ran out of time on this request.", "deadline_exceeded")

            try:
                response = await self.model.chat(ModelRequest(tuple(history), tool_specs))
            except Exception:
                await self._record(AuditEvent.now("error", request.context, "model error"))
                return AgentResponse.stopped(MODEL_ERROR_MESSAGE, "model_error")

            if response.has_tool_calls:
                history.append(Message.assistant(response.text, response.tool_calls))
                for call in response.tool_calls:
                    result = await self._invoke_tool(call.name, call.arguments, request, tool_by_name)
                    history.append(
                        Message.tool_result(
                            call.id,
                            call.name,
                            _tool_result_for_model(call.name, result, self.frame_tool_results),
                        )
                    )
                continue

            return AgentResponse.completed(response.text)

        return AgentResponse.stopped(MAX_STEPS_MESSAGE, "max_steps")

    async def _invoke_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        request: AgentRequest,
        tool_by_name: dict[str, Tool],
    ) -> ToolResult:
        tool = tool_by_name.get(name)
        if tool is None:
            await self._record(AuditEvent.now("tool.unavailable", request.context, f"tool={name}"))
            return ToolResult.failed(f"tool '{name}' is not available")

        validation = await self.argument_validator.validate(tool.spec, arguments, request.context)
        if not validation.allowed:
            await self._record(
                AuditEvent.now(
                    "tool.invalid_arguments",
                    request.context,
                    f"tool={name} reason={validation.reason}",
                )
            )
            return ToolResult.failed(validation.reason)

        decision = await self.tool_approver.approve(tool.spec, arguments, request.context)
        if not decision.allowed:
            await self._record(
                AuditEvent.now("tool.denied", request.context, f"tool={name} reason={decision.reason}")
            )
            return ToolResult.failed(decision.reason)

        await self._record(AuditEvent.now("tool.start", request.context, f"tool={name}"))
        try:
            result = await _invoke_with_timeout(
                tool,
                arguments,
                request,
                self.tool_timeout_seconds,
            )
            if result is None:
                await self._record(AuditEvent.now("tool.timeout", request.context, f"tool={name}"))
                return ToolResult.failed(f"tool '{name}' timed out")
        except Exception:
            await self._record(AuditEvent.now("tool.error", request.context, f"tool={name}"))
            return ToolResult.failed(f"tool '{name}' failed")
        await self._record(AuditEvent.now("tool.end", request.context, f"tool={name}"))
        return ToolResult(result.content[: self.max_tool_result_chars], result.error)

    async def _record(self, event: AuditEvent) -> None:
        try:
            await self.audit_sink.record(event)
        except Exception:
            return None


def _deadline_exceeded(request: AgentRequest) -> bool:
    deadline = request.context.deadline
    return deadline is not None and datetime.now(timezone.utc) >= deadline


async def _invoke_with_timeout(
    tool: Tool,
    arguments: dict[str, Any],
    request: AgentRequest,
    timeout_seconds: float | None,
) -> ToolResult | None:
    if timeout_seconds is None or timeout_seconds <= 0:
        return await tool.invoke(arguments, request.context)

    result: ToolResult | None = None
    with anyio.move_on_after(timeout_seconds) as scope:
        result = await tool.invoke(arguments, request.context)
    if scope.cancel_called:
        return None
    return result


def _tool_result_for_model(name: str, result: ToolResult, frame: bool) -> str:
    if not frame:
        return result.content
    status = "error" if result.error else "ok"
    return f"tool '{name}' result ({status}):\n{result.content}"
