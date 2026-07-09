from __future__ import annotations

import functools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any

import anyio

from python_ai_agents.core.agent import AgentRequest, AgentResponse
from python_ai_agents.core.audit import AuditEvent, AuditSink, NullAuditSink
from python_ai_agents.core.memory import (
    ConversationStore,
    InMemoryConversationStore,
    InMemoryMemory,
    Memory,
)
from python_ai_agents.core.model import Message, ModelPort, ModelRequest, Role, ToolCall
from python_ai_agents.core.observe import AgentObserver
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
    observers: list[AgentObserver] = field(default_factory=list)
    conversation_store: ConversationStore | None = None
    remember_conversation: bool = True
    tool_timeout_seconds: float | None = 30.0
    max_tool_result_chars: int = 8_000
    frame_tool_results: bool = True

    def __post_init__(self) -> None:
        # Build the default store once, at construction, so concurrent run() calls
        # on a shared agent don't each race to create (and split across) their own.
        if self.remember_conversation and self.conversation_store is None:
            self.conversation_store = InMemoryConversationStore()

    async def run(self, request: AgentRequest) -> AgentResponse:
        turn_start = perf_counter()
        await self._notify("on_turn_start", request.input)
        if not self.remember_conversation:
            memory = _new_turn_memory(self.system_prompt, request.input)
            response = await self._run_with_memory(request, memory, persist_response=False)
            await self._notify("on_turn_end", response, _duration_since(turn_start))
            return response

        assert self.conversation_store is not None  # built in __post_init__
        async with self.conversation_store.memory(
            request.context.tenant,
            request.context.session_id,
        ) as memory:
            if self.system_prompt and not _has_system_prompt(memory, self.system_prompt):
                memory.add(Message.system(self.system_prompt))
            memory.add(Message.user(request.input))
            response = await self._run_with_memory(request, memory, persist_response=True)
            await self._notify("on_turn_end", response, _duration_since(turn_start))
            return response

    async def run_tool(
        self, name: str, arguments: dict[str, Any], request: AgentRequest
    ) -> ToolResult:
        """Invoke one named tool through the full governed pipeline.

        Host UIs (e.g. a SQL or guided-query panel) use this to drive a specific
        tool directly. The call goes through the same argument validation,
        approval, timeout, result framing, and audit/observer notification as an
        in-turn tool call — so it is governed and audited like any agent step.
        """
        await self._notify("on_tool_call", ToolCall(name=name, arguments=arguments))
        tool_by_name = {tool.spec.name: tool for tool in self.tools}
        return await self._invoke_tool(name, arguments, request, tool_by_name)

    async def _run_with_memory(
        self,
        request: AgentRequest,
        memory: Memory,
        persist_response: bool,
    ) -> AgentResponse:
        active_tools = self.tool_selector.select(request.input, list(self.tools), request.context)
        tool_by_name = {tool.spec.name: tool for tool in active_tools}
        tool_specs = tuple(tool.spec for tool in active_tools)

        for _step in range(max(1, self.max_steps)):
            if _deadline_exceeded(request):
                return AgentResponse.stopped(
                    "I ran out of time on this request.",
                    "deadline_exceeded",
                )

            try:
                model_request = ModelRequest(memory.history(), tool_specs)
                await self._notify("on_model_call", model_request)
                model_start = perf_counter()
                response = await self.model.chat(model_request)
                model_latency = _duration_since(model_start)
                await self._notify("on_model_response", response, model_latency)
                await self._notify("on_usage", _model_name(self.model), response.usage)
            except Exception as exc:
                await self._record(AuditEvent.now("error", request.context, "model error"))
                await self._notify("on_error", "model", exc)
                return AgentResponse.stopped(MODEL_ERROR_MESSAGE, "model_error")

            if response.has_tool_calls:
                memory.add(Message.assistant(response.text, response.tool_calls))
                for call in response.tool_calls:
                    if _deadline_exceeded(request):
                        return AgentResponse.stopped(
                            "I ran out of time on this request.",
                            "deadline_exceeded",
                        )
                    await self._notify("on_tool_call", call)
                    result = await self._invoke_tool(
                        call.name,
                        call.arguments,
                        request,
                        tool_by_name,
                    )
                    memory.add(
                        Message.tool_result(
                            call.id,
                            call.name,
                            _tool_result_for_model(call.name, result, self.frame_tool_results),
                        )
                    )
                continue

            if persist_response:
                memory.add(Message.assistant(response.text))
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
            result = ToolResult.failed(f"tool '{name}' is not available")
            await self._notify("on_tool_result", name, result, timedelta(0))
            return result

        validation = await self.argument_validator.validate(tool.spec, arguments, request.context)
        if not validation.allowed:
            await self._record(
                AuditEvent.now(
                    "tool.invalid_arguments",
                    request.context,
                    f"tool={name} reason={validation.reason}",
                )
            )
            result = ToolResult.failed(validation.reason)
            await self._notify("on_tool_result", name, result, timedelta(0))
            return result

        decision = await self.tool_approver.approve(tool.spec, arguments, request.context)
        if not decision.allowed:
            await self._record(
                AuditEvent.now(
                    "tool.denied",
                    request.context,
                    f"tool={name} reason={decision.reason}",
                )
            )
            result = ToolResult.failed(decision.reason)
            await self._notify("on_tool_result", name, result, timedelta(0))
            return result

        await self._record(AuditEvent.now("tool.start", request.context, f"tool={name}"))
        try:
            tool_start = perf_counter()
            outcome = await _invoke_with_timeout(
                tool,
                arguments,
                request,
                self.tool_timeout_seconds,
            )
            if outcome is None:
                await self._record(AuditEvent.now("tool.timeout", request.context, f"tool={name}"))
                timed_out = ToolResult.failed(f"tool '{name}' timed out")
                await self._notify("on_tool_result", name, timed_out, _duration_since(tool_start))
                return timed_out
        except Exception:
            await self._record(AuditEvent.now("tool.error", request.context, f"tool={name}"))
            await self._notify("on_error", "tool", RuntimeError(f"tool '{name}' failed"))
            failed = ToolResult.failed(f"tool '{name}' failed")
            await self._notify("on_tool_result", name, failed, _duration_since(tool_start))
            return failed
        await self._record(AuditEvent.now("tool.end", request.context, f"tool={name}"))
        capped = ToolResult(
            outcome.content[: self.max_tool_result_chars],
            outcome.error,
            outcome.data,
            outcome.provenance,
            outcome.trust,
        )
        await self._notify("on_tool_result", name, capped, _duration_since(tool_start))
        return capped

    async def _record(self, event: AuditEvent) -> None:
        try:
            await self.audit_sink.record(event)
        except Exception:
            return None

    async def _notify(self, method: str, *args: Any) -> None:
        for observer in self.observers:
            try:
                callback = getattr(observer, method)
                await callback(*args)
            except Exception:
                continue


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

    # Run the tool in a worker thread so a blocking/CPU-bound tool can't pin the
    # event loop and IS abandoned when the timeout fires. Python cannot interrupt
    # the thread, so an abandoned tool finishes in the background — this bounds the
    # turn's latency, not the thread. Truly-async tools still work correctly.
    # ponytail: thread-per-timed-call; fine at tool-call rates, revisit if it's hot.
    result: ToolResult | None = None
    with anyio.move_on_after(timeout_seconds) as scope:
        result = await anyio.to_thread.run_sync(
            functools.partial(anyio.run, tool.invoke, arguments, request.context),
            abandon_on_cancel=True,
        )
    if scope.cancel_called:
        return None
    return result


def _trust_marker(trust: dict[str, Any] | None) -> str:
    """Machine-checkable trust-grade signal rendered into the model message.

    A tool that sets ``ToolResult.trust`` gets a ``[TRUST:TIER]`` token so the
    model can honor the grade programmatically (e.g. abstain from causal claims
    on ``[TRUST:INSUFFICIENT]``) rather than having to parse prose.
    """
    if not trust:
        return ""
    tier = trust.get("tier")
    if not tier:
        return ""
    if tier == "INSUFFICIENT":
        return "[TRUST:INSUFFICIENT] — insufficient evidence; do not assert causal or confident claims."
    return f"[TRUST:{tier}]"


def _tool_result_for_model(name: str, result: ToolResult, frame: bool) -> str:
    if not frame:
        body = result.content
    else:
        status = "error" if result.error else "ok"
        body = f"tool '{name}' result ({status}):\n{result.content}"
    marker = _trust_marker(result.trust)
    return f"{body}\n{marker}" if marker else body


def _new_turn_memory(system_prompt: str | None, input_text: str) -> Memory:
    memory = InMemoryMemory()
    if system_prompt:
        memory.add(Message.system(system_prompt))
    memory.add(Message.user(input_text))
    return memory


def _has_system_prompt(memory: Memory, system_prompt: str) -> bool:
    return any(
        message.role == Role.SYSTEM and message.content == system_prompt
        for message in memory.history()
    )


def _duration_since(start: float) -> timedelta:
    return timedelta(seconds=perf_counter() - start)


def _model_name(model: ModelPort) -> str:
    value = getattr(model, "model", None)
    if isinstance(value, str):
        return value
    return model.__class__.__name__
