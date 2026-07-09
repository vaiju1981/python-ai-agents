from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from python_ai_agents.core.context import RequestContext

if TYPE_CHECKING:
    from python_ai_agents.core.model import ToolCall


class ToolEffect(str, Enum):
    READ_ONLY = "read_only"
    EFFECTFUL = "effectful"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    effect: ToolEffect = ToolEffect.EFFECTFUL


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    error: bool = False
    data: Any = None  # optional structured payload (e.g. rows) for UIs; never sent to the model
    provenance: dict[str, Any] | None = None  # audit envelope: sql, dataset fingerprint, row count, timestamp, engine version

    @classmethod
    def ok(cls, content: str, data: Any = None, provenance: dict[str, Any] | None = None) -> ToolResult:
        return cls(content=content, data=data, provenance=provenance)

    @classmethod
    def failed(cls, content: str, provenance: dict[str, Any] | None = None) -> ToolResult:
        return cls(content=content, error=True, provenance=provenance)


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def invoke(self, arguments: dict[str, Any], context: RequestContext) -> ToolResult: ...


class ToolSelector(Protocol):
    def select(self, input_text: str, tools: list[Tool], context: RequestContext) -> list[Tool]: ...


class AllTools:
    def select(self, input_text: str, tools: list[Tool], context: RequestContext) -> list[Tool]:
        return list(tools)


@dataclass(frozen=True, slots=True)
class AllowListToolSelector:
    names: set[str]

    def select(self, input_text: str, tools: list[Tool], context: RequestContext) -> list[Tool]:
        return [tool for tool in tools if tool.spec.name in self.names]


@dataclass(frozen=True, slots=True)
class ToolDecision:
    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls) -> ToolDecision:
        return cls(True)

    @classmethod
    def deny(cls, reason: str) -> ToolDecision:
        return cls(False, reason)


class ToolApprover(Protocol):
    async def approve(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision: ...


class ToolArgumentValidator(Protocol):
    async def validate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision: ...


class NoopToolArgumentValidator:
    async def validate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        return ToolDecision.allow()


class RequiredArgumentsValidator:
    """Lightweight validator for JSON-schema-style required fields."""

    async def validate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        required = spec.input_schema.get("required", [])
        if not isinstance(required, list):
            return ToolDecision.deny(f"tool '{spec.name}' has invalid required-field metadata")
        missing = [name for name in required if isinstance(name, str) and name not in arguments]
        if missing:
            return ToolDecision.deny(
                f"tool '{spec.name}' missing required argument(s): {', '.join(missing)}"
            )
        return ToolDecision.allow()


class DenyEffectfulTools:
    async def approve(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        if spec.effect == ToolEffect.EFFECTFUL:
            return ToolDecision.deny(f"effectful tool '{spec.name}' requires approval")
        return ToolDecision.allow()


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A request to approve an effectful tool call that policy did not auto-approve.

    The ``approval_id`` is a unique token for this request — surfaced to a UI
    via ``AgentObserver`` so the decision can be resolved out of band.
    """

    approval_id: str
    call: ToolCall
    spec: ToolSpec
    principal: str
    tenant: str


class ApprovalHandler(Protocol):
    """Decides whether an effectful tool call may run, possibly by consulting a human.

    ``request_approval`` may block awaiting a human. Returning ``False``
    declines the call: the model is told the user declined and the turn
    continues.
    """

    async def request_approval(self, request: ApprovalRequest) -> bool: ...


@dataclass(slots=True)
class HumanApprovalToolApprover:
    """Escalates denied tool calls to an ``ApprovalHandler``.

    The wrapped policy still gets the first decision. If it allows the call,
    no human approval is requested. If it denies the call, the handler receives
    an ``ApprovalRequest`` and may override the denial by returning ``True``.
    """

    handler: ApprovalHandler
    policy: ToolApprover = DenyEffectfulTools()
    approval_id_factory: Callable[[], str] = lambda: str(uuid4())

    async def approve(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        from python_ai_agents.core.model import ToolCall

        decision = await self.policy.approve(spec, arguments, context)
        if decision.allowed:
            return decision

        request = ApprovalRequest(
            approval_id=self.approval_id_factory(),
            call=ToolCall(name=spec.name, arguments=arguments),
            spec=spec,
            principal=context.principal,
            tenant=context.tenant,
        )
        if await self.handler.request_approval(request):
            return ToolDecision.allow()
        return ToolDecision.deny("denied by the user")


class ConsoleToolApprover:
    """A human-in-the-loop ``ToolApprover``: prompts on stdin and permits on 'y'."""

    async def approve(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        import sys

        print(
            f"\n[approval] run {spec.effect.value} tool '{spec.name}'"
            f" with args {arguments}? [y/N] ",
            file=sys.stderr,
            flush=True,
        )
        try:
            line = input().strip().lower()
            return ToolDecision.allow() if line == "y" else ToolDecision.deny("denied by the user")
        except Exception as exc:
            return ToolDecision.deny(f"approval prompt failed: {exc}")
