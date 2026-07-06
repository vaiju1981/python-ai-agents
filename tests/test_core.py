from datetime import datetime, timedelta, timezone

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    Checkpoint,
    GuardrailDecision,
    GuardrailStage,
    SQLiteAuditSink,
    SQLiteCheckpointStore,
    ToolEffect,
    ToolResult,
    ToolSpec,
    Trust,
)


class EchoAgent:
    async def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse.completed(request.input)


class BlockInputGuardrail:
    async def check(self, stage, content, context):
        if stage == GuardrailStage.INPUT:
            return GuardrailDecision.block("blocked", "test_block")
        return GuardrailDecision.allow(content)


class EchoTool:
    def __init__(self, effect: ToolEffect) -> None:
        self._spec = ToolSpec(
            name="echo",
            description="Echoes the message argument.",
            input_schema={"type": "object"},
            effect=effect,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, arguments, context):
        return ToolResult.ok(str(arguments["message"]))


def test_govern_blocks_input() -> None:
    async def run() -> None:
        agent = Trust.govern(EchoAgent(), guardrails=[BlockInputGuardrail()])
        response = await agent.run(AgentRequest.ephemeral("hello"))

        assert response.blocked
        assert response.stop_reason == "test_block"

    anyio.run(run)


def test_govern_honors_expired_deadline() -> None:
    async def run() -> None:
        request = AgentRequest.ephemeral("hello")
        expired = datetime.now(timezone.utc) - timedelta(seconds=1)
        request = AgentRequest(request.input, request.context.__class__(
            session_id=request.context.session_id,
            trace_id=request.context.trace_id,
            deadline=expired,
        ))

        response = await Trust.govern(EchoAgent()).run(request)

        assert response.stop_reason == "deadline_exceeded"

    anyio.run(run)


def test_sqlite_checkpoint_store_round_trips(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteCheckpointStore(tmp_path / "runtime.sqlite3")

        await store.save(Checkpoint("tenant-a", "run-1", '{"step": 1}'))
        assert await store.load("tenant-a", "run-1") == Checkpoint(
            "tenant-a",
            "run-1",
            '{"step": 1}',
        )

        await store.save(Checkpoint("tenant-a", "run-1", '{"step": 2}'))
        assert await store.load("tenant-a", "run-1") == Checkpoint(
            "tenant-a",
            "run-1",
            '{"step": 2}',
        )

        await store.delete("tenant-a", "run-1")
        assert await store.load("tenant-a", "run-1") is None

    anyio.run(run)


def test_sqlite_audit_sink_records_governed_turn(tmp_path) -> None:
    async def run() -> None:
        audit = SQLiteAuditSink(tmp_path / "runtime.sqlite3")
        request = AgentRequest.ephemeral("hello")

        response = await Trust.govern(EchoAgent(), audit_sink=audit).run(request)

        assert response.output == "hello"
        events = audit.events(trace_id=request.context.trace_id)
        assert [event.event_type for event in events] == ["turn.start", "turn.end"]
        assert events[0].session_id == request.context.session_id
        assert events[-1].detail == "stopReason=completed"

    anyio.run(run)


def test_govern_tool_allows_read_only_tool(tmp_path) -> None:
    async def run() -> None:
        audit = SQLiteAuditSink(tmp_path / "runtime.sqlite3")
        context = AgentRequest.ephemeral("hello").context
        tool = Trust.govern_tool(EchoTool(ToolEffect.READ_ONLY), audit_sink=audit)

        result = await tool.invoke({"message": "hello"}, context)

        assert result == ToolResult.ok("hello")
        assert [event.event_type for event in audit.events(trace_id=context.trace_id)] == [
            "tool.start",
            "tool.end",
        ]

    anyio.run(run)


def test_govern_tool_denies_effectful_tool_by_default(tmp_path) -> None:
    async def run() -> None:
        audit = SQLiteAuditSink(tmp_path / "runtime.sqlite3")
        context = AgentRequest.ephemeral("hello").context
        tool = Trust.govern_tool(EchoTool(ToolEffect.EFFECTFUL), audit_sink=audit)

        result = await tool.invoke({"message": "hello"}, context)

        assert result.error
        assert "requires approval" in result.content
        assert [event.event_type for event in audit.events(trace_id=context.trace_id)] == [
            "tool.start",
            "tool.denied",
            "tool.end",
        ]

    anyio.run(run)
