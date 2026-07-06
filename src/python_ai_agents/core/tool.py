from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from python_ai_agents.core.context import RequestContext


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

    @classmethod
    def ok(cls, content: str) -> ToolResult:
        return cls(content=content)

    @classmethod
    def failed(cls, content: str) -> ToolResult:
        return cls(content=content, error=True)


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        ...

    async def invoke(self, arguments: dict[str, Any], context: RequestContext) -> ToolResult:
        ...


class ToolSelector(Protocol):
    def select(self, input_text: str, tools: list[Tool], context: RequestContext) -> list[Tool]:
        ...


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
    ) -> ToolDecision:
        ...


class ToolArgumentValidator(Protocol):
    async def validate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        ...


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
