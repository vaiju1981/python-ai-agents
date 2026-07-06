from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import anyio

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.audit import AuditEvent, AuditSink, NullAuditSink
from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.guardrail import Guardrail, GuardrailDecision, GuardrailStage
from python_ai_agents.core.idempotency import (
    IdempotencyStore,
    IdempotentAgent,
    InMemoryIdempotencyStore,
)
from python_ai_agents.core.tool import (
    DenyEffectfulTools,
    Tool,
    ToolApprover,
    ToolResult,
    ToolSpec,
)


DEADLINE_MESSAGE = "I ran out of time on this request."


@dataclass(slots=True)
class GovernedAgent:
    delegate: Agent
    guardrails: list[Guardrail] = field(default_factory=list)
    audit_sink: AuditSink = field(default_factory=NullAuditSink)

    async def run(self, request: AgentRequest) -> AgentResponse:
        ctx = request.context
        await self._audit("turn.start", request, f"input.len={len(request.input)}")
        end_reason = "error"
        try:
            if _deadline_exceeded(request):
                end_reason = "deadline_exceeded"
                return AgentResponse.stopped(DEADLINE_MESSAGE, "deadline_exceeded")

            remaining = _remaining_seconds(request)
            if remaining is None:
                response = await self._governed_turn(request)
            else:
                with anyio.move_on_after(remaining) as scope:
                    response = await self._governed_turn(request)
                if scope.cancel_called:
                    end_reason = "deadline_exceeded"
                    return AgentResponse.stopped(DEADLINE_MESSAGE, "deadline_exceeded")

            end_reason = "blocked" if response.blocked else f"stopReason={response.stop_reason}"
            return response
        finally:
            await self._record(AuditEvent.now("turn.end", ctx, end_reason))

    async def _governed_turn(self, request: AgentRequest) -> AgentResponse:
        in_decision = await self._apply_guardrails(GuardrailStage.INPUT, request.input, request)
        if in_decision.blocked:
            await self._audit("guardrail.block", request, f"stage=INPUT reason={in_decision.reason}")
            return AgentResponse.blocked_response(in_decision.content, in_decision.reason)

        response = await self.delegate.run(AgentRequest(in_decision.content, request.context))

        out_decision = await self._apply_guardrails(GuardrailStage.OUTPUT, response.output, request)
        if out_decision.blocked:
            await self._audit(
                "guardrail.block",
                request,
                f"stage=OUTPUT reason={out_decision.reason}",
            )
            return AgentResponse.blocked_response(out_decision.content, out_decision.reason)

        return AgentResponse(out_decision.content, response.blocked, response.stop_reason)

    async def _apply_guardrails(
        self,
        stage: GuardrailStage,
        content: str,
        request: AgentRequest,
    ) -> GuardrailDecision:
        current = content
        for guardrail in self.guardrails:
            decision = await guardrail.check(stage, current, request.context)
            if decision.blocked:
                return decision
            current = decision.content
        return GuardrailDecision.allow(current)

    async def _audit(self, event_type: str, request: AgentRequest, detail: str) -> None:
        await self._record(AuditEvent.now(event_type, request.context, detail))

    async def _record(self, event: AuditEvent) -> None:
        try:
            await self.audit_sink.record(event)
        except Exception:
            # Audit must never break an agent run. A concrete logger can be added later.
            return None


@dataclass(slots=True)
class GovernedTool:
    delegate: Tool
    approver: ToolApprover = field(default_factory=DenyEffectfulTools)
    audit_sink: AuditSink = field(default_factory=NullAuditSink)

    @property
    def spec(self) -> ToolSpec:
        return self.delegate.spec

    async def invoke(self, arguments: dict[str, Any], context: RequestContext) -> ToolResult:
        await self._record(
            AuditEvent.now(
                "tool.start",
                context,
                f"name={self.spec.name} effect={self.spec.effect.value}",
            )
        )
        end_reason = "error"
        try:
            decision = await self.approver.approve(self.spec, arguments, context)
            if not decision.allowed:
                end_reason = "denied"
                await self._record(
                    AuditEvent.now(
                        "tool.denied",
                        context,
                        f"name={self.spec.name} reason={decision.reason}",
                    )
                )
                return ToolResult.failed(decision.reason)

            result = await self.delegate.invoke(arguments, context)
            end_reason = "error" if result.error else "completed"
            return result
        finally:
            await self._record(AuditEvent.now("tool.end", context, end_reason))

    async def _record(self, event: AuditEvent) -> None:
        try:
            await self.audit_sink.record(event)
        except Exception:
            return None


class Trust:
    @staticmethod
    def govern(
        agent: Agent,
        guardrails: list[Guardrail] | None = None,
        audit_sink: AuditSink | None = None,
    ) -> Agent:
        return GovernedAgent(agent, guardrails or [], audit_sink or NullAuditSink())

    @staticmethod
    def govern_tool(
        tool: Tool,
        approver: ToolApprover | None = None,
        audit_sink: AuditSink | None = None,
    ) -> Tool:
        return GovernedTool(tool, approver or DenyEffectfulTools(), audit_sink or NullAuditSink())

    @staticmethod
    def idempotent(
        agent: Agent,
        store: IdempotencyStore | None = None,
        key_attribute: str = "idempotencyKey",
    ) -> Agent:
        return IdempotentAgent(agent, store or InMemoryIdempotencyStore(), key_attribute)


def _deadline_exceeded(request: AgentRequest) -> bool:
    deadline = request.context.deadline
    return deadline is not None and datetime.now(timezone.utc) >= deadline


def _remaining_seconds(request: AgentRequest) -> float | None:
    deadline = request.context.deadline
    if deadline is None:
        return None
    return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
