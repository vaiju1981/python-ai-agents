from typing import Any

import anyio

from python_ai_agents import (
    AgentRequest,
    DefaultAgent,
    ModelRequest,
    ModelResponse,
    RecordingObserver,
    RedactingObserver,
    TokenAccountingObserver,
    ToolCall,
    ToolEffect,
    ToolResult,
    ToolSpec,
    Usage,
)


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse] | None = None, fail: bool = False) -> None:
        self.responses = list(responses or [])
        self.fail = fail

    async def chat(self, request: ModelRequest) -> ModelResponse:
        if self.fail:
            raise RuntimeError("model failed")
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse.text_response("ok", Usage(input_tokens=2, output_tokens=3))


class EchoTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echoes the message argument.",
            input_schema={"type": "object"},
            effect=ToolEffect.READ_ONLY,
        )

    async def invoke(self, arguments: dict[str, Any], context) -> ToolResult:
        return ToolResult.ok(str(arguments["message"]))


class FailingObserver:
    async def on_turn_start(self, input_text: str) -> None:
        raise RuntimeError("observer failed")


def test_recording_observer_records_model_turn() -> None:
    async def run() -> None:
        observer = RecordingObserver()
        response = await DefaultAgent(ScriptedModel(), observers=[observer]).run(
            AgentRequest.ephemeral("hi")
        )

        assert response.output == "ok"
        assert observer.turn_inputs == ["hi"]
        assert observer.model_requests[0].messages[-1].content == "hi"
        assert observer.model_responses == [ModelResponse.text_response("ok", Usage(2, 3))]
        assert observer.turn_responses == [response]

    anyio.run(run)


def test_recording_observer_records_tool_calls_and_results() -> None:
    async def run() -> None:
        observer = RecordingObserver()
        call = ToolCall(name="echo", arguments={"message": "pong"}, id="call-1")
        model = ScriptedModel(
            [
                ModelResponse.tool_response((call,)),
                ModelResponse.text_response("done"),
            ]
        )

        await DefaultAgent(model, tools=[EchoTool()], observers=[observer]).run(
            AgentRequest.ephemeral("hi")
        )

        assert observer.tool_calls == [call]
        assert observer.tool_results == [ToolResult.ok("pong")]

    anyio.run(run)


def test_recording_observer_records_model_errors_and_turn_end() -> None:
    async def run() -> None:
        observer = RecordingObserver()
        response = await DefaultAgent(ScriptedModel(fail=True), observers=[observer]).run(
            AgentRequest.ephemeral("hi")
        )

        assert response.stop_reason == "model_error"
        assert observer.errors == [("model", "RuntimeError")]
        assert observer.turn_responses == [response]

    anyio.run(run)


def test_redacting_observer_forwards_metadata_without_content() -> None:
    async def run() -> None:
        recorder = RecordingObserver()
        observer = RedactingObserver(recorder)
        call = ToolCall(name="echo", arguments={"message": "secret"}, id="call-1")
        model = ScriptedModel(
            [
                ModelResponse.tool_response((call,), text="secret", usage=Usage(total_tokens=7)),
                ModelResponse.text_response("final secret"),
            ]
        )

        await DefaultAgent(model, tools=[EchoTool()], observers=[observer]).run(
            AgentRequest.ephemeral("user secret")
        )

        assert recorder.turn_inputs == ["[redacted]"]
        assert recorder.model_requests[0].messages[-1].content == "[redacted]"
        assert recorder.model_responses[0].text == "[redacted]"
        assert recorder.model_responses[0].tool_calls[0].arguments == {"redacted": "[redacted]"}
        assert recorder.tool_results == [ToolResult.ok("[redacted]")]

    anyio.run(run)


def test_token_accounting_observer_sums_usage() -> None:
    async def run() -> None:
        observer = TokenAccountingObserver()

        await DefaultAgent(ScriptedModel(), observers=[observer]).run(AgentRequest.ephemeral("hi"))

        assert observer.model_calls == 1
        assert observer.input_tokens == 2
        assert observer.output_tokens == 3
        assert observer.total_tokens == 5

    anyio.run(run)


def test_observer_failures_do_not_break_turn() -> None:
    async def run() -> None:
        response = await DefaultAgent(ScriptedModel(), observers=[FailingObserver()]).run(
            AgentRequest.ephemeral("hi")
        )

        assert response.output == "ok"

    anyio.run(run)
