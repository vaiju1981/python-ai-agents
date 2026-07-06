from datetime import datetime, timedelta, timezone

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    Checkpoint,
    GuardrailDecision,
    GuardrailStage,
    InMemoryAuditSink,
    InMemoryCheckpointStore,
    InMemoryIdempotencyStore,
    RequestContext,
    SQLiteAuditSink,
    SQLiteCheckpointStore,
    StopCategory,
    StopReason,
    ToolEffect,
    ToolResult,
    ToolSpec,
    Trust,
)


class EchoAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.calls += 1
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


def test_response_stop_reason_classification() -> None:
    assert AgentResponse.completed("ok").is_completed
    assert AgentResponse.completed("ok").reason == StopReason.COMPLETED
    assert AgentResponse.stopped("later", "deadline_exceeded").retryable
    assert AgentResponse.stopped("later", "deadline_exceeded").reason.category == StopCategory.TIMEOUT
    assert AgentResponse.blocked_response("blocked", "policy").reason == StopReason.BLOCKED
    assert not AgentResponse.stopped("too much", "budget_exceeded").retryable


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


def test_in_memory_checkpoint_store_evicts_oldest() -> None:
    async def run() -> None:
        store = InMemoryCheckpointStore(max_entries=1)

        await store.save(Checkpoint("tenant-a", "run-1", "{}"))
        await store.save(Checkpoint("tenant-a", "run-2", "{}"))

        assert await store.load("tenant-a", "run-1") is None
        assert await store.load("tenant-a", "run-2") == Checkpoint("tenant-a", "run-2", "{}")

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


def test_in_memory_audit_sink_filters_events() -> None:
    async def run() -> None:
        audit = InMemoryAuditSink()
        request = AgentRequest.ephemeral("hello")

        await Trust.govern(EchoAgent(), audit_sink=audit).run(request)

        assert [event.event_type for event in audit.events(trace_id=request.context.trace_id)] == [
            "turn.start",
            "turn.end",
        ]
        assert audit.events(session_id="missing") == []

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


def test_idempotent_agent_replays_non_retryable_response() -> None:
    async def run() -> None:
        delegate = EchoAgent()
        agent = Trust.idempotent(delegate, InMemoryIdempotencyStore())
        context = RequestContext(
            session_id="session-1",
            principal="user-1",
            tenant="tenant-a",
            attributes={"idempotencyKey": "request-1"},
        )

        first = await agent.run(AgentRequest("first", context))
        second = await agent.run(AgentRequest("second", context))

        assert first == AgentResponse.completed("first")
        assert second == first
        assert delegate.calls == 1

    anyio.run(run)


def test_idempotent_agent_does_not_cache_retryable_response() -> None:
    class FlakyAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, request: AgentRequest) -> AgentResponse:
            self.calls += 1
            if self.calls == 1:
                return AgentResponse.stopped("try again", "model_error")
            return AgentResponse.completed(request.input)

    async def run() -> None:
        delegate = FlakyAgent()
        agent = Trust.idempotent(delegate, InMemoryIdempotencyStore())
        context = RequestContext(
            session_id="session-1",
            principal="user-1",
            tenant="tenant-a",
            attributes={"idempotencyKey": "request-1"},
        )

        first = await agent.run(AgentRequest("first", context))
        second = await agent.run(AgentRequest("second", context))

        assert first.retryable
        assert second == AgentResponse.completed("second")
        assert delegate.calls == 2

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
