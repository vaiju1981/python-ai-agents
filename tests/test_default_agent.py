from typing import Any

import anyio

from python_ai_agents import (
    AgentRequest,
    AllowListToolSelector,
    DefaultAgent,
    InMemoryAuditSink,
    Message,
    ModelRequest,
    ModelResponse,
    RecordingObserver,
    RequiredArgumentsValidator,
    ToolCall,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from python_ai_agents.adapters import JsonSchemaToolArgumentValidator


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
    def __init__(
        self,
        effect: ToolEffect = ToolEffect.READ_ONLY,
        name: str = "echo",
        input_schema: dict[str, Any] | None = None,
    ) -> None:
        self._spec = ToolSpec(
            name=name,
            description="Echoes the message argument.",
            input_schema=input_schema or {"type": "object"},
            effect=effect,
        )
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, arguments, context):
        self.calls.append(arguments)
        return ToolResult.ok(str(arguments["message"]))


class SlowTool(EchoTool):
    async def invoke(self, arguments, context):
        await anyio.sleep(0.2)
        return ToolResult.ok("late")


class BlockingTool(EchoTool):
    """A blocking/CPU tool: sleeps synchronously, with no await point to cancel at."""

    async def invoke(self, arguments, context):
        import time

        time.sleep(0.5)  # cannot be cancelled cooperatively
        return ToolResult.ok("late")


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
        assert second_request_messages[-1] == Message.tool_result(
            "call-1",
            "echo",
            "tool 'echo' result (ok):\npong",
        )
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


def test_default_agent_only_exposes_selected_tools() -> None:
    async def run() -> None:
        model = ScriptedModel([ModelResponse.text_response("done")])
        agent = DefaultAgent(
            model,
            tools=[EchoTool(name="echo"), EchoTool(name="hidden")],
            tool_selector=AllowListToolSelector({"echo"}),
        )

        await agent.run(AgentRequest.ephemeral("ping"))

        assert [tool.name for tool in model.requests[0].tools] == ["echo"]

    anyio.run(run)


def test_default_agent_rejects_unselected_tool_call() -> None:
    async def run() -> None:
        model = ScriptedModel(
            [
                ModelResponse.tool_response(
                    (ToolCall(name="hidden", arguments={"message": "secret"}, id="call-1"),)
                ),
                ModelResponse.text_response("handled"),
            ]
        )
        audit = InMemoryAuditSink()
        hidden = EchoTool(name="hidden")
        agent = DefaultAgent(
            model,
            tools=[EchoTool(name="echo"), hidden],
            tool_selector=AllowListToolSelector({"echo"}),
            audit_sink=audit,
        )

        response = await agent.run(AgentRequest.ephemeral("ping"))

        assert response.output == "handled"
        assert hidden.calls == []
        assert "not available" in model.requests[1].messages[-1].content
        assert [event.event_type for event in audit.events()] == ["tool.unavailable"]

    anyio.run(run)


def test_default_agent_validates_required_arguments() -> None:
    async def run() -> None:
        model = ScriptedModel(
            [
                ModelResponse.tool_response((ToolCall(name="echo", arguments={}, id="call-1"),)),
                ModelResponse.text_response("handled"),
            ]
        )
        audit = InMemoryAuditSink()
        tool = EchoTool(input_schema={"type": "object", "required": ["message"]})
        agent = DefaultAgent(
            model,
            tools=[tool],
            argument_validator=RequiredArgumentsValidator(),
            audit_sink=audit,
        )

        response = await agent.run(AgentRequest.ephemeral("ping"))

        assert response.output == "handled"
        assert tool.calls == []
        assert "missing required argument" in model.requests[1].messages[-1].content
        assert [event.event_type for event in audit.events()] == ["tool.invalid_arguments"]

    anyio.run(run)


def test_default_agent_validates_arguments_with_jsonschema_adapter() -> None:
    async def run() -> None:
        model = ScriptedModel(
            [
                ModelResponse.tool_response(
                    (ToolCall(name="echo", arguments={"message": 123}, id="call-1"),)
                ),
                ModelResponse.text_response("handled"),
            ]
        )
        tool = EchoTool(
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            }
        )
        agent = DefaultAgent(
            model,
            tools=[tool],
            argument_validator=JsonSchemaToolArgumentValidator(),
        )

        await agent.run(AgentRequest.ephemeral("ping"))

        assert tool.calls == []
        assert "schema validation" in model.requests[1].messages[-1].content

    anyio.run(run)


def test_default_agent_times_out_tool_call() -> None:
    async def run() -> None:
        model = ScriptedModel(
            [
                ModelResponse.tool_response(
                    (ToolCall(name="echo", arguments={"message": "slow"}, id="call-1"),)
                ),
                ModelResponse.text_response("handled"),
            ]
        )
        audit = InMemoryAuditSink()
        agent = DefaultAgent(
            model,
            tools=[SlowTool()],
            audit_sink=audit,
            tool_timeout_seconds=0.01,
        )

        await agent.run(AgentRequest.ephemeral("ping"))

        assert "timed out" in model.requests[1].messages[-1].content
        assert [event.event_type for event in audit.events()] == ["tool.start", "tool.timeout"]

    anyio.run(run)


def test_default_agent_times_out_blocking_sync_tool() -> None:
    # Regression: a blocking (non-async) tool body must still honor the timeout.
    async def run() -> None:
        import time

        model = ScriptedModel(
            [
                ModelResponse.tool_response(
                    (ToolCall(name="echo", arguments={"message": "slow"}, id="call-1"),)
                ),
                ModelResponse.text_response("handled"),
            ]
        )
        audit = InMemoryAuditSink()
        agent = DefaultAgent(
            model,
            tools=[BlockingTool()],
            audit_sink=audit,
            tool_timeout_seconds=0.02,
        )

        started = time.perf_counter()
        await agent.run(AgentRequest.ephemeral("ping"))
        elapsed = time.perf_counter() - started

        assert "timed out" in model.requests[1].messages[-1].content
        assert [event.event_type for event in audit.events()] == ["tool.start", "tool.timeout"]
        assert elapsed < 0.4  # bounded well under the tool's 0.5s block

    anyio.run(run)


def test_default_agent_caps_tool_result_before_model_context() -> None:
    async def run() -> None:
        model = ScriptedModel(
            [
                ModelResponse.tool_response(
                    (ToolCall(name="echo", arguments={"message": "abcdef"}, id="call-1"),)
                ),
                ModelResponse.text_response("handled"),
            ]
        )
        agent = DefaultAgent(
            model, tools=[EchoTool()], max_tool_result_chars=3, frame_tool_results=False
        )

        await agent.run(AgentRequest.ephemeral("ping"))

        assert model.requests[1].messages[-1] == Message.tool_result("call-1", "echo", "abc")

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


def test_default_agent_run_tool_goes_through_pipeline_and_audit() -> None:
    async def run() -> None:
        audit = InMemoryAuditSink()
        observer = RecordingObserver()
        tool = EchoTool(name="echo")
        agent = DefaultAgent(
            ScriptedModel(),
            tools=[tool],
            audit_sink=audit,
            observers=[observer],
        )

        result = await agent.run_tool(
            "echo", {"message": "hi"}, AgentRequest.ephemeral("host call")
        )

        assert not result.error
        assert result.content == "hi"
        # Tool actually executed...
        assert tool.calls == [{"message": "hi"}]
        # ...and the call was governed + audited like an in-turn step.
        assert [event.event_type for event in audit.events()] == ["tool.start", "tool.end"]
        assert [c.name for c in observer.tool_calls] == ["echo"]

    anyio.run(run)


def test_default_agent_run_tool_unknown_tool_fails_without_audit() -> None:
    async def run() -> None:
        audit = InMemoryAuditSink()
        agent = DefaultAgent(ScriptedModel(), tools=[EchoTool(name="echo")], audit_sink=audit)

        result = await agent.run_tool("missing", {}, AgentRequest.ephemeral("x"))

        assert result.error
        assert [event.event_type for event in audit.events()] == ["tool.unavailable"]

    anyio.run(run)
