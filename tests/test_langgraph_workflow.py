"""Tests for the LangGraph-backed recoverable workflow adapter."""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from typing import Any

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    InMemoryAuditSink,
    InMemoryCheckpointStore,
    RequestContext,
    SQLiteCheckpointStore,
    StopReason,
)
from python_ai_agents.adapters import (
    LangGraphAgent,
    StoreCheckpointSaver,
    WorkflowState,
    agent_node,
    recoverable_agent,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class EchoAgent:
    """Simple Agent that echoes input with a prefix."""

    def __init__(self, prefix: str = "echo") -> None:
        self.prefix = prefix
        self.calls: list[str] = []

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.calls.append(request.input)
        return AgentResponse.completed(f"{self.prefix}: {request.input}")


# ---------------------------------------------------------------------------
# StoreCheckpointSaver
# ---------------------------------------------------------------------------


def test_store_checkpoint_saver_round_trips_through_in_memory_store() -> None:
    async def run() -> None:
        from langgraph.graph import END, START, StateGraph

        store = InMemoryCheckpointStore()
        saver = StoreCheckpointSaver(store)

        builder = StateGraph(WorkflowState, context_schema=RequestContext)
        builder.add_node("agent", agent_node(EchoAgent()))
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        graph = builder.compile(checkpointer=saver)

        ctx = RequestContext.session("thread-1")
        result = await graph.ainvoke(
            {"input": "hello"},
            config={"configurable": {"thread_id": "thread-1", "tenant": ctx.tenant}},
            context=ctx,
        )
        assert result["output"] == "echo: hello"

        # The checkpoint should be in our store.
        ckpt = await store.load(ctx.tenant, "thread-1")
        assert ckpt is not None
        assert "checkpoint_id" in ckpt.payload_json

    anyio.run(run)


def test_store_checkpoint_saver_persists_to_sqlite() -> None:
    async def run() -> None:
        import tempfile
        from pathlib import Path

        from langgraph.graph import END, START, StateGraph

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteCheckpointStore(Path(tmp) / "checkpoints.db")
            saver = StoreCheckpointSaver(store)

            builder = StateGraph(WorkflowState, context_schema=RequestContext)
            builder.add_node("agent", agent_node(EchoAgent("sqlite")))
            builder.add_edge(START, "agent")
            builder.add_edge("agent", END)
            graph = builder.compile(checkpointer=saver)

            ctx = RequestContext.session("sqlite-thread")
            await graph.ainvoke(
                {"input": "persisted"},
                config={"configurable": {"thread_id": "sqlite-thread", "tenant": ctx.tenant}},
                context=ctx,
            )

            ckpt = await store.load(ctx.tenant, "sqlite-thread")
            assert ckpt is not None

    anyio.run(run)


# ---------------------------------------------------------------------------
# recoverable_agent – single-agent checkpointed workflow
# ---------------------------------------------------------------------------


def test_recoverable_agent_runs_and_completes() -> None:
    async def run() -> None:
        echo = EchoAgent()
        store = InMemoryCheckpointStore()
        agent = recoverable_agent(echo, checkpoint_store=store)

        ctx = RequestContext.session("s1")
        response = await agent.run(AgentRequest("hello", ctx))

        assert response.output == "echo: hello"
        assert response.is_completed
        assert echo.calls == ["hello"]

    anyio.run(run)


def test_recoverable_agent_persists_checkpoint() -> None:
    async def run() -> None:
        store = InMemoryCheckpointStore()
        agent = recoverable_agent(EchoAgent(), checkpoint_store=store)

        ctx = RequestContext.session("persist-1")
        await agent.run(AgentRequest("world", ctx))

        ckpt = await store.load(ctx.tenant, "persist-1")
        assert ckpt is not None

    anyio.run(run)


def test_recoverable_agent_records_audit() -> None:
    async def run() -> None:
        audit = InMemoryAuditSink()
        store = InMemoryCheckpointStore()
        agent = recoverable_agent(EchoAgent(), checkpoint_store=store, audit_sink=audit)

        ctx = RequestContext.session("audit-1")
        await agent.run(AgentRequest("audited", ctx))

        events = audit.events(session_id="audit-1")
        types = [e.event_type for e in events]
        assert "turn.start" in types
        assert "turn.end" in types

    anyio.run(run)


# ---------------------------------------------------------------------------
# Multi-node workflow with interrupt / resume
# ---------------------------------------------------------------------------


def test_multi_node_workflow_completes() -> None:
    async def run() -> None:
        from langgraph.graph import END, START, StateGraph

        calls: list[str] = []

        async def step1(state: Any, *, runtime: Any) -> dict[str, str]:
            calls.append("step1")
            return {"output": state["input"] + " -> step1"}

        async def step2(state: Any, *, runtime: Any) -> dict[str, str]:
            calls.append("step2")
            return {"output": state["output"] + " -> step2"}

        store = InMemoryCheckpointStore()
        saver = StoreCheckpointSaver(store)
        builder = StateGraph(WorkflowState, context_schema=RequestContext)
        builder.add_node("step1", step1)
        builder.add_node("step2", step2)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", END)
        graph = builder.compile(checkpointer=saver)

        agent = LangGraphAgent(graph)
        ctx = RequestContext.session("multi-1")
        response = await agent.run(AgentRequest("data", ctx))

        assert response.is_completed
        assert response.output == "data -> step1 -> step2"
        assert calls == ["step1", "step2"]

    anyio.run(run)


def test_interrupt_before_pauses_and_resume_continues() -> None:
    async def run() -> None:
        from langgraph.graph import END, START, StateGraph

        calls: list[str] = []

        async def step1(state: Any, *, runtime: Any) -> dict[str, str]:
            calls.append("step1")
            return {"output": state["input"] + " -> step1"}

        async def step2(state: Any, *, runtime: Any) -> dict[str, str]:
            calls.append("step2")
            return {"output": state["output"] + " -> step2"}

        store = InMemoryCheckpointStore()
        saver = StoreCheckpointSaver(store)
        builder = StateGraph(WorkflowState, context_schema=RequestContext)
        builder.add_node("step1", step1)
        builder.add_node("step2", step2)
        builder.add_edge(START, "step1")
        builder.add_edge("step1", "step2")
        builder.add_edge("step2", END)
        graph = builder.compile(checkpointer=saver, interrupt_before=["step2"])

        agent = LangGraphAgent(graph)
        ctx = RequestContext.session("interrupt-1")

        # First call – should pause before step2.
        r1 = await agent.run(AgentRequest("data", ctx))
        assert not r1.is_completed
        assert r1.stop_reason == "interrupted"
        assert r1.reason == StopReason.INTERRUPTED
        assert r1.reason.category.value == "incomplete"
        assert calls == ["step1"]

        # Resume with empty input.
        r2 = await agent.run(AgentRequest("", ctx))
        assert r2.is_completed
        assert r2.output == "data -> step1 -> step2"
        assert calls == ["step1", "step2"]

    anyio.run(run)


def test_interrupt_persists_checkpoint_across_saver_instances() -> None:
    """A crash between the first and second call should be recoverable."""
    async def run() -> None:
        import tempfile
        from pathlib import Path

        from langgraph.graph import END, START, StateGraph

        calls: list[str] = []

        async def step1(state: Any, *, runtime: Any) -> dict[str, str]:
            calls.append("step1")
            return {"output": state["input"] + " -> step1"}

        async def step2(state: Any, *, runtime: Any) -> dict[str, str]:
            calls.append("step2")
            return {"output": state["output"] + " -> step2"}

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "recover.db"
            store = SQLiteCheckpointStore(db_path)

            saver1 = StoreCheckpointSaver(store)
            builder = StateGraph(WorkflowState, context_schema=RequestContext)
            builder.add_node("step1", step1)
            builder.add_node("step2", step2)
            builder.add_edge(START, "step1")
            builder.add_edge("step1", "step2")
            builder.add_edge("step2", END)
            graph1 = builder.compile(checkpointer=saver1, interrupt_before=["step2"])

            ctx = RequestContext.session("crash-1")
            agent1 = LangGraphAgent(graph1)
            r1 = await agent1.run(AgentRequest("persisted", ctx))
            assert r1.stop_reason == "interrupted"
            assert calls == ["step1"]

            # Simulate a restart: new store and saver pointing at the same DB.
            store2 = SQLiteCheckpointStore(db_path)
            saver2 = StoreCheckpointSaver(store2)
            graph2 = builder.compile(checkpointer=saver2, interrupt_before=["step2"])
            agent2 = LangGraphAgent(graph2)

            r2 = await agent2.run(AgentRequest("", ctx))
            assert r2.is_completed
            assert r2.output == "persisted -> step1 -> step2"
            assert calls == ["step1", "step2"]

    anyio.run(run)


# ---------------------------------------------------------------------------
# agent_node
# ---------------------------------------------------------------------------


def test_agent_node_passes_request_context() -> None:
    async def run() -> None:
        from langgraph.graph import END, START, StateGraph

        captured: list[RequestContext] = []

        class CapturingAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                captured.append(request.context)
                return AgentResponse.completed(f"{request.context.principal}: {request.input}")

        store = InMemoryCheckpointStore()
        saver = StoreCheckpointSaver(store)
        builder = StateGraph(WorkflowState, context_schema=RequestContext)
        builder.add_node("agent", agent_node(CapturingAgent()))
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        graph = builder.compile(checkpointer=saver)

        agent = LangGraphAgent(graph)
        ctx = RequestContext(session_id="ctx-1", principal="alice", tenant="acme")
        response = await agent.run(AgentRequest("hello", ctx))

        assert response.output == "alice: hello"
        assert captured[0].principal == "alice"
        assert captured[0].tenant == "acme"

    anyio.run(run)


def test_agent_node_with_custom_keys() -> None:
    async def run() -> None:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class CustomState(TypedDict):
            query: str
            answer: str

        store = InMemoryCheckpointStore()
        saver = StoreCheckpointSaver(store)

        class QAAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed(f"A: {request.input}")

        builder = StateGraph(CustomState, context_schema=RequestContext)
        builder.add_node("qa", agent_node(QAAgent(), input_key="query", output_key="answer"))
        builder.add_edge(START, "qa")
        builder.add_edge("qa", END)
        graph = builder.compile(checkpointer=saver)

        agent = LangGraphAgent(graph, input_key="query", output_key="answer")
        ctx = RequestContext.session("custom-1")
        response = await agent.run(AgentRequest("What is 2+2?", ctx))

        assert response.output == "A: What is 2+2?"

    anyio.run(run)


# ---------------------------------------------------------------------------
# LangGraphAgent with Trust integration
# ---------------------------------------------------------------------------


def test_langgraph_agent_works_behind_trust_guard() -> None:
    async def run() -> None:
        from python_ai_agents import GuardrailDecision, GuardrailStage, Trust

        class BlockGuardrail:
            async def check(self, stage, content, context):
                if stage == GuardrailStage.INPUT and "blocked" in content:
                    return GuardrailDecision.block("blocked", "test_block")
                return GuardrailDecision.allow(content)

        store = InMemoryCheckpointStore()
        inner = recoverable_agent(EchoAgent(), checkpoint_store=store)
        agent = Trust.govern(inner, guardrails=[BlockGuardrail()])

        ctx = RequestContext.session("trust-1")

        # Blocked input
        r1 = await agent.run(AgentRequest("blocked input", ctx))
        assert r1.blocked
        assert r1.stop_reason == "test_block"

        # Allowed input
        r2 = await agent.run(AgentRequest("ok input", ctx))
        assert r2.is_completed
        assert r2.output == "echo: ok input"

    anyio.run(run)
