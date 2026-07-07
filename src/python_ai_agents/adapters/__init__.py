"""Optional substrate adapters."""

from typing import Any

from python_ai_agents.adapters.ollama import (
    DEFAULT_OLLAMA_TEST_MODELS,
    RECOMMENDED_SAMPLING,
    OllamaAgent,
    OllamaError,
    OllamaHttpTransport,
    OllamaModelPort,
)

__all__ = [
    "DEFAULT_OLLAMA_TEST_MODELS",
    "RECOMMENDED_SAMPLING",
    "DeepEvalScorer",
    "GuardrailsAiGuardrail",
    "McpToolAdapter",
    "OtelAgentObserver",
    "JsonSchemaToolArgumentValidator",
    "LangGraphAgent",
    "OllamaAgent",
    "OllamaError",
    "OllamaHttpTransport",
    "OllamaModelPort",
    "PresidioScrubGuardrail",
    "StoreCheckpointSaver",
    "WorkflowState",
    "agent_node",
    "recoverable_agent",
]


def __getattr__(name: str) -> Any:
    if name == "JsonSchemaToolArgumentValidator":
        from python_ai_agents.adapters.jsonschema_tools import JsonSchemaToolArgumentValidator

        return JsonSchemaToolArgumentValidator
    if name in (
        "LangGraphAgent",
        "StoreCheckpointSaver",
        "WorkflowState",
        "agent_node",
        "recoverable_agent",
    ):
        from python_ai_agents.adapters import langgraph as _lg

        return getattr(_lg, name)
    if name == "PresidioScrubGuardrail":
        from python_ai_agents.adapters.presidio_guardrails import PresidioScrubGuardrail

        return PresidioScrubGuardrail
    if name == "GuardrailsAiGuardrail":
        from python_ai_agents.adapters.guardrails_ai import GuardrailsAiGuardrail

        return GuardrailsAiGuardrail
    if name == "DeepEvalScorer":
        from python_ai_agents.adapters.deepeval import DeepEvalScorer

        return DeepEvalScorer
    if name == "McpToolAdapter":
        from python_ai_agents.adapters.mcp import McpToolAdapter

        return McpToolAdapter
    if name == "OtelAgentObserver":
        from python_ai_agents.adapters.otel import OtelAgentObserver

        return OtelAgentObserver
    raise AttributeError(name)
