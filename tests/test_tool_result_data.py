"""ToolResult.data (structured tool output) flows through the loop to observers."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    DefaultAgent,
    ModelResponse,
    NoopAgentObserver,
    ToolCall,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class _DataTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="rows",
            description="",
            input_schema={"type": "object"},
            effect=ToolEffect.READ_ONLY,
        )

    async def invoke(self, arguments, context):
        return ToolResult.ok("[rows — text for the model]", data=[{"a": 1}, {"a": 2}])


class _ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, request):
        return self.responses.pop(0) if self.responses else ModelResponse.text_response("done")


class _Capture(NoopAgentObserver):
    def __init__(self):
        self.data = None

    async def on_tool_result(self, tool_name, result, latency):
        self.data = result.data


def test_tool_result_data_reaches_observer_through_capping() -> None:
    async def run() -> None:
        model = _ScriptedModel(
            [
                ModelResponse.tool_response((ToolCall(name="rows", arguments={}, id="c1"),)),
                ModelResponse.text_response("done"),
            ]
        )
        capture = _Capture()
        agent = DefaultAgent(model, tools=[_DataTool()], observers=[capture])

        await agent.run(AgentRequest.ephemeral("go"))

        # Structured data survives result-capping and reaches the observer exactly.
        assert capture.data == [{"a": 1}, {"a": 2}]

    anyio.run(run)
