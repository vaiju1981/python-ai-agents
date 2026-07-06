"""LangGraph-backed recoverable workflow adapter.

This module provides a thin adapter that wraps LangGraph's graph engine with our
``RequestContext``, audit, trust, and checkpoint/conversation stores.  It is an
optional extra (``pip install python-ai-agents[langgraph]``).

Building blocks
---------------

* ``StoreCheckpointSaver`` – bridges our ``CheckpointStore`` to LangGraph's
  ``BaseCheckpointSaver`` so that graph checkpoints are persisted in our durable
  stores.
* ``LangGraphAgent`` – wraps a compiled LangGraph graph as our ``Agent``
  protocol, mapping ``RequestContext`` to a LangGraph thread config and
  recording audit events.
* ``agent_node`` – converts any ``Agent`` into a LangGraph node function that
  receives ``RequestContext`` via the graph's ``context_schema``.
* ``recoverable_agent`` – convenience builder for a single-agent checkpointed
  workflow.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse
from python_ai_agents.core.audit import AuditEvent, AuditSink, NullAuditSink
from python_ai_agents.core.checkpoint import Checkpoint as StoredCheckpoint
from python_ai_agents.core.checkpoint import CheckpointStore
from python_ai_agents.core.context import RequestContext

__all__ = [
    "LangGraphAgent",
    "StoreCheckpointSaver",
    "WorkflowState",
    "agent_node",
    "recoverable_agent",
]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class WorkflowState(TypedDict):
    """Default state schema for recoverable workflows.

    Nodes read ``input`` and write ``output``.  Custom workflows may use a
    different ``TypedDict`` as long as the same keys are supplied to
    ``LangGraphAgent``.
    """

    input: str
    output: str


# ---------------------------------------------------------------------------
# Checkpoint saver bridge
# ---------------------------------------------------------------------------


class StoreCheckpointSaver(BaseCheckpointSaver[Any]):
    """Bridges our ``CheckpointStore`` to LangGraph's ``BaseCheckpointSaver``.

    LangGraph checkpoints (including channel blobs and pending writes) are
    serialised into a single JSON document stored via our ``CheckpointStore``,
    keyed by ``(tenant, thread_id)``.  Only the latest checkpoint per thread is
    retained, which is sufficient for crash-recovery and interrupt-resume.
    """

    def __init__(self, store: CheckpointStore) -> None:
        super().__init__(serde=JsonPlusSerializer())
        self.store = store

    # -- async API (used by compiled graphs via ainvoke) -------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, tenant = _config_keys(config)
        stored = await self.store.load(tenant, thread_id)
        if stored is None:
            return None
        doc = json.loads(stored.payload_json)
        requested_id = get_checkpoint_id(config)
        if requested_id is not None and requested_id != doc["checkpoint_id"]:
            return None
        return _to_tuple(doc, thread_id, self.serde)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, tenant = _config_keys(config)
        ckpt = dict(checkpoint)
        channel_values: dict[str, Any] = cast(dict[str, Any], ckpt.pop("channel_values", {}))

        # Preserve blobs from prior checkpoints for channels whose version
        # did not change in this superstep.
        existing = await self.store.load(tenant, thread_id)
        blobs: dict[str, dict[str, list[str]]] = {}
        if existing is not None:
            blobs = dict(json.loads(existing.payload_json).get("blobs", {}))

        for channel, version in new_versions.items():
            if channel in channel_values:
                blobs.setdefault(channel, {})[str(version)] = _dumps_typed(
                    self.serde, channel_values[channel]
                )
            else:
                blobs.setdefault(channel, {})[str(version)] = ["empty", ""]

        doc: dict[str, Any] = {
            "checkpoint_id": checkpoint["id"],
            "checkpoint_typed": _dumps_typed(self.serde, ckpt),
            "metadata_typed": _dumps_typed(
                self.serde, get_checkpoint_metadata(config, metadata)
            ),
            "parent_checkpoint_id": get_checkpoint_id(config),
            "channel_versions": dict(checkpoint.get("channel_versions", {})),
            "blobs": blobs,
            "writes": {},
        }
        payload = json.dumps(doc, separators=(",", ":"))
        await self.store.save(
            StoredCheckpoint(tenant=tenant, run_id=thread_id, payload_json=payload)
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "tenant": tenant,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, tenant = _config_keys(config)
        existing = await self.store.load(tenant, thread_id)
        if existing is None:
            return
        doc = json.loads(existing.payload_json)
        writes_map: dict[str, list[Any]] = doc.setdefault("writes", {})
        for _idx, (channel, value) in enumerate(writes):
            key = f"{task_id}\x00{channel}"
            if channel in WRITES_IDX_MAP:
                writes_map[key] = [task_id, channel, _dumps_typed(self.serde, value), task_path]
            elif key not in writes_map:
                writes_map[key] = [task_id, channel, _dumps_typed(self.serde, value), task_path]
        await self.store.save(
            StoredCheckpoint(
                tenant=tenant,
                run_id=thread_id,
                payload_json=json.dumps(doc, separators=(",", ":")),
            )
        )

    async def alist(  # type: ignore[override]
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            return
        thread_id, tenant = _config_keys(config)
        stored = await self.store.load(tenant, thread_id)
        if stored is None:
            return
        doc = json.loads(stored.payload_json)
        yield _to_tuple(doc, thread_id, self.serde)

# ---------------------------------------------------------------------------
# Agent ↔ LangGraph node bridge
# ---------------------------------------------------------------------------


def agent_node(
    delegate: Agent,
    *,
    input_key: str = "input",
    output_key: str = "output",
) -> Callable[..., Any]:
    """Wrap an ``Agent`` as a LangGraph node function.

    The returned coroutine expects ``runtime.context`` to be a
    ``RequestContext`` (set via the graph's ``context_schema``).
    """

    async def node(state: Any, *, runtime: Any) -> dict[str, Any]:
        context: RequestContext = runtime.context
        text = _state_get(state, input_key, "")
        response = await delegate.run(AgentRequest(input=text, context=context))
        return {output_key: response.output}

    return node


# ---------------------------------------------------------------------------
# LangGraphAgent – wraps a compiled graph as our Agent protocol
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LangGraphAgent:
    """Wraps a compiled LangGraph graph as our ``Agent`` protocol.

    Maps ``RequestContext`` to a LangGraph thread config (``thread_id`` =
    session id, ``tenant`` = tenant), records audit events for turn
    start/end, and detects interrupts so callers can resume.
    """

    graph: Any
    audit_sink: AuditSink = field(default_factory=NullAuditSink)
    input_key: str = "input"
    output_key: str = "output"

    async def run(self, request: AgentRequest) -> AgentResponse:
        ctx = request.context
        config: RunnableConfig = {
            "configurable": {
                "thread_id": ctx.session_id,
                "tenant": ctx.tenant,
            }
        }
        await self._audit("turn.start", ctx, f"input.len={len(request.input)}")
        end_reason = "error"
        try:
            state = await self.graph.aget_state(config=config)
            if state.next and not request.input:
                # Resume from an interrupt.
                result = await self.graph.ainvoke(None, config=config, context=ctx)
            else:
                result = await self.graph.ainvoke(
                    {self.input_key: request.input},
                    config=config,
                    context=ctx,
                )

            output = _extract_output(result, self.output_key)
            new_state = await self.graph.aget_state(config=config)
            if new_state.next:
                end_reason = "interrupted"
                return AgentResponse.stopped(output, "interrupted")
            end_reason = "completed"
            return AgentResponse.completed(output)
        except Exception as exc:
            await self._audit("turn.error", ctx, f"error={exc.__class__.__name__}")
            return AgentResponse.stopped(
                "I ran into a problem with the workflow.", "model_error"
            )
        finally:
            await self._audit("turn.end", ctx, end_reason)

    async def _audit(self, event_type: str, ctx: RequestContext, detail: str) -> None:
        try:
            await self.audit_sink.record(AuditEvent.now(event_type, ctx, detail))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def recoverable_agent(
    delegate: Agent,
    checkpoint_store: CheckpointStore,
    *,
    audit_sink: AuditSink | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
) -> LangGraphAgent:
    """Wrap a single ``Agent`` in a checkpointed LangGraph workflow.

    The resulting ``LangGraphAgent`` persists its state after each turn via
    ``checkpoint_store`` and can be resumed after a crash or interrupt.
    """
    builder = StateGraph(WorkflowState, context_schema=RequestContext)
    builder.add_node("agent", agent_node(delegate))
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile(
        checkpointer=StoreCheckpointSaver(checkpoint_store),
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
    )
    return LangGraphAgent(graph, audit_sink=audit_sink or NullAuditSink())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _config_keys(config: RunnableConfig) -> tuple[str, str]:
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "")
    tenant = configurable.get("tenant", "default")
    return thread_id, tenant


def _dumps_typed(serde: Any, obj: Any) -> list[str]:
    type_str, raw = serde.dumps_typed(obj)
    return [type_str, base64.b64encode(raw).decode("ascii")]


def _loads_typed(serde: Any, typed: list[str]) -> Any:
    type_str, b64 = typed
    return serde.loads_typed((type_str, base64.b64decode(b64)))


def _to_tuple(
    doc: dict[str, Any],
    thread_id: str,
    serde: Any,
) -> CheckpointTuple:
    ckpt: dict[str, Any] = _loads_typed(serde, doc["checkpoint_typed"])
    metadata = _loads_typed(serde, doc["metadata_typed"])
    channel_versions: dict[str, Any] = doc.get("channel_versions", {})

    channel_values: dict[str, Any] = {}
    blobs = doc.get("blobs", {})
    for channel, version in channel_versions.items():
        channel_blobs = blobs.get(channel, {})
        typed = channel_blobs.get(str(version))
        if typed is not None and typed[0] != "empty":
            channel_values[channel] = _loads_typed(serde, typed)

    checkpoint: Checkpoint = ckpt.copy()  # type: ignore[assignment]
    checkpoint["channel_values"] = channel_values

    parent_id = doc.get("parent_checkpoint_id")
    parent_config: RunnableConfig | None = None
    if parent_id:
        parent_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": parent_id,
            }
        }

    writes_map: dict[str, list[Any]] = doc.get("writes", {})
    pending_writes: list[tuple[str, str, Any]] = [
        (entry[0], entry[1], _loads_typed(serde, entry[2]))
        for entry in writes_map.values()
    ]

    return CheckpointTuple(
        config={
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": doc["checkpoint_id"],
            }
        },
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )


def _state_get(state: Any, key: str, default: Any) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _extract_output(result: Any, output_key: str) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        value = result.get(output_key, "")
    else:
        value = getattr(result, output_key, "")
    return str(value) if value is not None else ""
