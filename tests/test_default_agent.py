from typing import Any

import anyio

from python_ai_agents import (
    AgentRequest,
    DefaultAgent,
    InMemoryAuditSink,
    Message,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse] | None = None, fail: bool = False) -> None:
        self.responses = list(responses or [])
        self.requests: list[ModelRequest] = []
        self.fail = fail

    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("model exploded")
        if not self.responses:
            return ModelResponse.text_response("done")
        return self.responses.pop(0)


class EchoTool:
    def __init__(self, effect: ToolEffect = ToolEffect.READ_ONLY) -> None:
        self._spec = ToolSpec(
            name="echo",
            description="Echoes the message argument.",
            input_schema={"type": "object"},
            effect=effect,
        )
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, arguments, context):
        self.calls.append(arguments)
        return ToolResult.ok(str(arguments["message"]))


def test_default_agent_returns_model_text() -> None:
    async def run() -> None:
        model = ScriptedModel([ModelResponse.text_response("hello")])
        agent = DefaultAgent(model, system_prompt="Be terse.")

        response = await agent.run(AgentRequest.ephemeral("hi"))

        assert response.output == "hello"
        assert model.requests[0].messages == (
            Message.system("Be terse."),
            Message.user("hi"),
        )

    anyio.run(run)


def test_default_agent_executes_tool_then_finishes() -> None:
    async def run() -> None:
        tool_call = ToolCall(name="echo", arguments={"message": "pong"}, id="call-1")
        model = ScriptedModel(
            [
                ModelResponse.tool_response((tool_call,), text="I need a tool."),
                ModelResponse.text_response("tool said pong"),
            ]
        )
        audit = InMemoryAuditSink()
        tool = EchoTool()
        agent = DefaultAgent(model, tools=[tool], audit_sink=audit)

        response = await agent.run(AgentRequest.ephemeral("ping"))

        assert response.output == "tool said pong"
        assert tool.calls == [{"message": "pong"}]
        second_request_messages = model.requests[1].messages
        assert second_request_messages[-1] == Message.tool_result("call-1", "echo", "pong")
        assert [event.event_type for event in audit.events()] == ["tool.start", "tool.end"]

    anyio.run(run)


def test_default_agent_denies_effectful_tool_by_default() -> None:
    async def run() -> None:
        tool_call = ToolCall(name="echo", arguments={"message": "pong"}, id="call-1")
        model = ScriptedModel(
            [
                ModelResponse.tool_response((tool_call,)),
                ModelResponse.text_response("denied handled"),
            ]
        )
        audit = InMemoryAuditSink()
        tool = EchoTool(ToolEffect.EFFECTFUL)
        agent = DefaultAgent(model, tools=[tool], audit_sink=audit)

        response = await agent.run(AgentRequest.ephemeral("ping"))

        assert response.output == "denied handled"
        assert tool.calls == []
        assert "requires approval" in model.requests[1].messages[-1].content
        assert [event.event_type for event in audit.events()] == ["tool.denied"]

    anyio.run(run)


def test_default_agent_stops_on_model_error() -> None:
    async def run() -> None:
        audit = InMemoryAuditSink()
        response = await DefaultAgent(ScriptedModel(fail=True), audit_sink=audit).run(
            AgentRequest.ephemeral("hi")
        )

        assert response.stop_reason == "model_error"
        assert response.retryable
        assert [event.event_type for event in audit.events()] == ["error"]

    anyio.run(run)


def test_default_agent_stops_at_max_steps() -> None:
    async def run() -> None:
        model = ScriptedModel(
            [ModelResponse.tool_response((ToolCall(name="missing"),)) for _ in range(3)]
        )
        response = await DefaultAgent(model, max_steps=2).run(AgentRequest.ephemeral("hi"))

        assert response.stop_reason == "max_steps"
        assert len(model.requests) == 2

    anyio.run(run)
