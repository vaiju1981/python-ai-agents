"""Optional substrate adapters."""

from python_ai_agents.adapters.ollama import (
    DEFAULT_OLLAMA_TEST_MODELS,
    OllamaAgent,
    OllamaError,
    OllamaHttpTransport,
    OllamaModelPort,
)

__all__ = [
    "DEFAULT_OLLAMA_TEST_MODELS",
    "JsonSchemaToolArgumentValidator",
    "OllamaAgent",
    "OllamaError",
    "OllamaHttpTransport",
    "OllamaModelPort",
]


def __getattr__(name: str):
    if name == "JsonSchemaToolArgumentValidator":
        from python_ai_agents.adapters.jsonschema_tools import JsonSchemaToolArgumentValidator

        return JsonSchemaToolArgumentValidator
    raise AttributeError(name)
