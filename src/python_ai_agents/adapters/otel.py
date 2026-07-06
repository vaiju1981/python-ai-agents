"""OpenTelemetry observer adapter.

Wraps the ``opentelemetry-api`` library to emit spans and metrics for agent
observer events. Install::

    pip install python-ai-agents[opentelemetry]

Usage::

    from python_ai_agents.adapters.otel import OtelAgentObserver

    observer = OtelAgentObserver(service_name="my-agent")
    agent = DefaultAgent(model, observers=[observer])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from python_ai_agents.core.agent import AgentResponse
from python_ai_agents.core.model import ModelRequest, ModelResponse, Usage
from python_ai_agents.core.observe import NoopAgentObserver
from python_ai_agents.core.model import ToolCall
from python_ai_agents.core.tool import ToolResult

__all__ = ["OtelAgentObserver"]


@dataclass(slots=True)
class OtelAgentObserver(NoopAgentObserver):
    """Emits OpenTelemetry spans and metrics for agent events.

    Uses ``opentelemetry-api``'s tracer and meter. Observer failures are
    isolated — they never break an agent run.
    """

    service_name: str = "python-ai-agents"
    _tracer: Any = field(default=None, init=False)
    _meter: Any = field(default=None, init=False)
    _current_span: Any = field(default=None, init=False)
    _token_counter: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        try:
            from opentelemetry import trace, metrics

            self._tracer = trace.get_tracer(self.service_name)
            self._meter = metrics.get_meter(self.service_name)
            self._token_counter = self._meter.create_counter(
                "agent.tokens", description="Token usage by model"
            )
        except ImportError:
            pass

    async def on_turn_start(self, input_text: str) -> None:
        if self._tracer is None:
            return
        self._current_span = self._tracer.start_span("agent.turn")

    async def on_turn_end(self, response: AgentResponse, duration: timedelta) -> None:
        if self._current_span is not None:
            self._current_span.set_attribute("agent.stop_reason", response.stop_reason)
            self._current_span.set_attribute("agent.blocked", response.blocked)
            self._current_span.set_attribute("agent.duration_ms", duration.total_seconds() * 1000)
            self._current_span.end()
            self._current_span = None

    async def on_usage(self, model: str, usage: Usage) -> None:
        if self._token_counter is not None:
            self._token_counter.add(
                (usage.input_tokens or 0) + (usage.output_tokens or 0),
                {"model": model, "type": "total"},
            )

    async def on_error(self, stage: str, error: BaseException) -> None:
        if self._current_span is not None:
            self._current_span.record_exception(error)
            self._current_span.set_attribute("agent.error_stage", stage)
