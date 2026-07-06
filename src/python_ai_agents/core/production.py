"""Opinionated production assembly for the governed runtime.

Requires durable/session storage, audit delivery, and argument validation.
Enables model/tool timeouts, deny-effectful authorization, result
framing/capping, and redacted telemetry. Places one hard-deadline policy
boundary around the finished agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.audit import AuditSink, NullAuditSink
from python_ai_agents.core.default_agent import DefaultAgent
from python_ai_agents.core.guardrail import Guardrail
from python_ai_agents.core.memory import ConversationStore
from python_ai_agents.core.model import ModelPort
from python_ai_agents.core.observe import AgentObserver, RedactingObserver
from python_ai_agents.core.resilient import ResilientModelPort
from python_ai_agents.core.tool import (
    ApprovalHandler,
    DenyEffectfulTools,
    HumanApprovalToolApprover,
    Tool,
    ToolApprover,
    ToolArgumentValidator,
    NoopToolArgumentValidator,
)
from python_ai_agents.core.trust import Trust

__all__ = ["ProductionAgentRuntime"]


@dataclass(slots=True)
class ProductionAgentRuntime:
    """Production-ready agent assembly with safe defaults.

    Wraps a ``ResilientModelPort`` (retry + timeout) around the model, builds a
    ``DefaultAgent`` with durable storage, audit, validation, tool timeouts,
    result framing/capping, redacted observers, and an outer ``Trust.govern``
    boundary for the hard deadline and lifecycle audit.
    """

    delegate: Agent

    async def run(self, request: AgentRequest) -> AgentResponse:
        return await self.delegate.run(request)

    @staticmethod
    def builder() -> "_Builder":
        return _Builder()


@dataclass
class _Builder:
    model: ModelPort | None = None
    conversation_store: ConversationStore | None = None
    audit_sink: AuditSink | None = None
    argument_validator: ToolArgumentValidator = field(default_factory=NoopToolArgumentValidator)
    tool_approver: ToolApprover = field(default_factory=DenyEffectfulTools)
    approval_handler: ApprovalHandler | None = None
    tools: list[Tool] = field(default_factory=list)
    guardrails: list[Guardrail] = field(default_factory=list)
    observers: list[AgentObserver] = field(default_factory=list)
    raw_observers: list[AgentObserver] = field(default_factory=list)
    system_prompt: str | None = None
    model_timeout: float = 60.0
    model_attempts: int = 3
    model_backoff_ms: float = 500.0
    tool_timeout_seconds: float | None = 30.0
    max_tool_result_chars: int = 8_000
    max_steps: int = 8

    def with_model(self, model: ModelPort) -> "_Builder":
        self.model = model
        return self

    def with_conversation_store(self, store: ConversationStore) -> "_Builder":
        self.conversation_store = store
        return self

    def with_audit_sink(self, sink: AuditSink) -> "_Builder":
        self.audit_sink = sink
        return self

    def with_argument_validator(self, validator: ToolArgumentValidator) -> "_Builder":
        self.argument_validator = validator
        return self

    def with_tool_approver(self, approver: ToolApprover) -> "_Builder":
        self.tool_approver = approver
        return self

    def with_approval_handler(self, handler: ApprovalHandler) -> "_Builder":
        self.approval_handler = handler
        return self

    def tool(self, tool: Tool) -> "_Builder":
        self.tools.append(tool)
        return self

    def guardrail(self, guardrail: Guardrail) -> "_Builder":
        self.guardrails.append(guardrail)
        return self

    def observer(self, observer: AgentObserver) -> "_Builder":
        self.observers.append(observer)
        return self

    def raw_observer(self, observer: AgentObserver) -> "_Builder":
        self.raw_observers.append(observer)
        return self

    def with_system_prompt(self, prompt: str) -> "_Builder":
        self.system_prompt = prompt
        return self

    def build(self) -> ProductionAgentRuntime:
        if self.model is None:
            raise ValueError("model is required")
        if self.conversation_store is None:
            raise ValueError("conversation_store is required")
        if self.audit_sink is None:
            raise ValueError("audit_sink is required")

        resilient = ResilientModelPort(
            delegate=self.model,
            max_attempts=self.model_attempts,
            timeout_seconds=self.model_timeout,
            backoff_ms=self.model_backoff_ms,
        )

        all_observers: list[AgentObserver] = []
        for obs in self.observers:
            all_observers.append(RedactingObserver(obs))
        all_observers.extend(self.raw_observers)

        core = DefaultAgent(
            model=resilient,
            tools=list(self.tools),
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            audit_sink=self.audit_sink,
            observers=all_observers,
            conversation_store=self.conversation_store,
            tool_timeout_seconds=self.tool_timeout_seconds,
            max_tool_result_chars=self.max_tool_result_chars,
            frame_tool_results=True,
            tool_approver=self._tool_approver(),
            argument_validator=self.argument_validator,
        )

        governed = Trust.govern(core, guardrails=list(self.guardrails), audit_sink=self.audit_sink)
        return ProductionAgentRuntime(delegate=governed)

    def _tool_approver(self) -> ToolApprover:
        if self.approval_handler is None:
            return self.tool_approver
        return HumanApprovalToolApprover(
            handler=self.approval_handler,
            policy=self.tool_approver,
        )
