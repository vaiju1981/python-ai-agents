from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from python_ai_agents.core.agent import AgentRequest, AgentResponse
from python_ai_agents.core.audit import AuditEvent, AuditSink, NullAuditSink
from python_ai_agents.core.model import Message, ModelPort, ModelRequest, ModelResponse
from python_ai_agents.core.tool import DenyEffectfulTools, Tool, ToolApprover, ToolResult


MAX_STEPS_MESSAGE = "I couldn't finish that within my step budget. Please try rephrasing."
MODEL_ERROR_MESSAGE = "I ran into a problem reaching the model. Please try again."


@dataclass(slots=True)
class DefaultAgent:
    """Small model/tool loop over the ModelPort seam."""

    model: ModelPort
    tools: list[Tool] = field(default_factory=list)
    system_prompt: str | None = None
    max_steps: int = 8
    tool_approver: ToolApprover = field(default_factory=DenyEffectfulTools)
    audit_sink: AuditSink = field(default_factory=NullAuditSink)
    max_tool_result_chars: int = 8_000

    async def run(self, request: AgentRequest) -> AgentResponse:
        history: list[Message] = []
        if self.system_prompt:
            history.append(Message.system(self.system_prompt))
        history.append(Message.user(request.input))

        tool_by_name = {tool.spec.name: tool for tool in self.tools}
        tool_specs = tuple(tool.spec for tool in self.tools)

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
                    history.append(Message.tool_result(call.id, call.name, result.content))
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
            return ToolResult.failed(f"tool '{name}' is not available")

        decision = await self.tool_approver.approve(tool.spec, arguments, request.context)
        if not decision.allowed:
            await self._record(
                AuditEvent.now("tool.denied", request.context, f"tool={name} reason={decision.reason}")
            )
            return ToolResult.failed(decision.reason)

        await self._record(AuditEvent.now("tool.start", request.context, f"tool={name}"))
        try:
            result = await tool.invoke(arguments, request.context)
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
