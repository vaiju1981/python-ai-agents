"""Tests for episodic memory and replay tool executor."""

from __future__ import annotations

import anyio

from python_ai_agents import Episode, InMemoryEpisodicStore, ReplayToolExecutor, ToolResult


def test_episode_defaults_tenant() -> None:
    ep = Episode(tenant="", task="test", outcome="ok", success=True, lesson="")
    assert ep.tenant == "default"


def test_episodic_store_records_and_recalls() -> None:
    async def run() -> None:
        store = InMemoryEpisodicStore()
        await store.record(Episode("acme", "fix bug", "failed", False, "check logs"))
        await store.record(Episode("acme", "deploy app", "deployed", True, ""))
        await store.record(Episode("other", "fix bug", "ok", True, ""))

        results = await store.recall("acme", "fix bug", 10)
        assert len(results) == 1
        assert results[0].task == "fix bug"

        # Tenant isolation
        other = await store.recall("other", "fix bug", 10)
        assert len(other) == 1
        assert other[0].tenant == "other"

    anyio.run(run)


def test_episodic_store_keyword_matching() -> None:
    async def run() -> None:
        store = InMemoryEpisodicStore()
        await store.record(Episode("t", "analyze sales data", "ok", True, ""))
        await store.record(Episode("t", "deploy to prod", "ok", True, ""))

        results = await store.recall("t", "sales", 10)
        assert len(results) == 1
        assert "sales" in results[0].task

    anyio.run(run)


def test_replay_tool_executor_returns_recorded() -> None:
    executor = ReplayToolExecutor(
        recorded={
            ("echo", '{"message":"hello"}'): ToolResult.ok("hello"),
        }
    )
    result = executor.execute("echo", {"message": "hello"})
    assert not result.error
    assert result.content == "hello"


def test_replay_tool_executor_fallback_order() -> None:
    executor = ReplayToolExecutor(
        recorded={
            ("a", '{"x":1}'): ToolResult.ok("a1"),
            ("b", '{"y":2}'): ToolResult.ok("b2"),
        }
    )
    # Unknown key → falls back to insertion order
    r1 = executor.execute("unknown", {"z": 0})
    assert r1.content == "a1"
    r2 = executor.execute("unknown", {"z": 0})
    assert r2.content == "b2"


def test_replay_tool_executor_exhausted() -> None:
    executor = ReplayToolExecutor(recorded={})
    result = executor.execute("missing", {})
    assert result.error
    assert "replay exhausted" in result.content
