"""Tests for agents-as-tools, approval, and production runtime."""

from __future__ import annotations

import anyio
import pytest

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    ApprovalRequest,
    HumanApprovalToolApprover,
    InMemoryAuditSink,
    InMemoryConversationStore,
    ModelRequest,
    ModelResponse,
    ProductionAgentRuntime,
    RequestContext,
    ToolCall,
    ToolEffect,
    ToolResult,
    ToolSpec,
    agent_as_tool,
)


class ScriptedModel:
    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls = 0

    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse.text_response(self.text)


class EchoAgent:
    async def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse.completed(f"echo: {request.input}")


def test_agent_as_tool_wraps_agent() -> None:
    async def run() -> None:
        tool = agent_as_tool(
            "research",
            "gathers facts",
            EchoAgent(),
            effect=ToolEffect.READ_ONLY,
        )
        assert tool.spec.name == "research"
        assert tool.spec.effect == ToolEffect.READ_ONLY
        ctx = RequestContext.ephemeral()
        result = await tool.invoke({"request": "find X"}, ctx)
        assert not result.error
        assert "echo: find X" in result.content

    anyio.run(run)


def test_agent_as_tool_passes_context() -> None:
    async def run() -> None:
        class CtxAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed(
                    f"{request.context.principal}:{request.context.tenant}"
                )

        tool = agent_as_tool("ctx", "test", CtxAgent(), effect=ToolEffect.READ_ONLY)
        ctx = RequestContext(session_id="s1", principal="alice", tenant="acme")
        result = await tool.invoke({"request": "test"}, ctx)
        assert "alice:acme" in result.content

    anyio.run(run)


def test_agent_as_tool_blocked_returns_error() -> None:
    async def run() -> None:
        class BlockingAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.blocked_response("blocked", "policy")

        tool = agent_as_tool("blocked", "test", BlockingAgent())
        result = await tool.invoke({"request": "test"}, RequestContext.ephemeral())
        assert result.error

    anyio.run(run)


def test_production_agent_runtime_builds() -> None:
    async def run() -> None:
        model = ScriptedModel("prod ok")
        runtime = (
            ProductionAgentRuntime.builder()
            .with_model(model)
            .with_conversation_store(InMemoryConversationStore())
            .with_audit_sink(InMemoryAuditSink())
            .with_system_prompt("Reply tersely.")
            .build()
        )
        response = await runtime.run(AgentRequest.ephemeral("hi"))
        assert response.output.strip()

    anyio.run(run)


def test_production_agent_runtime_requires_store() -> None:
    with pytest.raises(ValueError, match="conversation_store"):
        ProductionAgentRuntime.builder().with_model(ScriptedModel()).build()


def test_approval_request_is_frozen() -> None:
    from python_ai_agents import ToolCall, ToolSpec

    req = ApprovalRequest(
        approval_id="id1",
        call=ToolCall(name="tool", arguments={}),
        spec=ToolSpec(name="tool", description="test", input_schema={}),
        principal="alice",
        tenant="acme",
    )
    assert req.approval_id == "id1"
    assert req.principal == "alice"


def test_human_approval_tool_approver_escalates_denied_tool() -> None:
    async def run() -> None:
        class Handler:
            def __init__(self) -> None:
                self.requests: list[ApprovalRequest] = []

            async def request_approval(self, request: ApprovalRequest) -> bool:
                self.requests.append(request)
                return True

        handler = Handler()
        approver = HumanApprovalToolApprover(
            handler=handler,
            approval_id_factory=lambda: "approval-1",
        )
        spec = ToolSpec(
            name="write_file",
            description="writes a file",
            input_schema={"type": "object"},
            effect=ToolEffect.EFFECTFUL,
        )
        decision = await approver.approve(
            spec,
            {"path": "out.txt"},
            RequestContext(session_id="s1", principal="alice", tenant="acme"),
        )
        assert decision.allowed
        assert handler.requests[0].approval_id == "approval-1"
        assert handler.requests[0].call.name == "write_file"
        assert handler.requests[0].principal == "alice"

    anyio.run(run)


def test_production_runtime_uses_approval_handler_for_effectful_tools() -> None:
    async def run() -> None:
        class SequencedModel:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, request: ModelRequest) -> ModelResponse:
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse.tool_response(
                        (ToolCall(name="write_file", arguments={"path": "out.txt"}),)
                    )
                return ModelResponse.text_response("done")

        class WriteTool:
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="write_file",
                    description="writes a file",
                    input_schema={"type": "object"},
                    effect=ToolEffect.EFFECTFUL,
                )

            async def invoke(self, arguments, context):
                return ToolResult.ok("wrote")

        class Handler:
            def __init__(self) -> None:
                self.calls = 0

            async def request_approval(self, request: ApprovalRequest) -> bool:
                self.calls += 1
                return True

        handler = Handler()
        runtime = (
            ProductionAgentRuntime.builder()
            .with_model(SequencedModel())
            .with_conversation_store(InMemoryConversationStore())
            .with_audit_sink(InMemoryAuditSink())
            .with_approval_handler(handler)
            .tool(WriteTool())
            .build()
        )
        response = await runtime.run(AgentRequest.ephemeral("write it"))
        assert response.output == "done"
        assert handler.calls == 1

    anyio.run(run)
