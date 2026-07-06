"""Optional substrate adapters."""

from python_ai_agents.adapters.ollama import (
    DEFAULT_OLLAMA_TEST_MODELS,
    OllamaAgent,
    OllamaError,
    OllamaHttpTransport,
)

__all__ = [
    "DEFAULT_OLLAMA_TEST_MODELS",
    "OllamaAgent",
    "OllamaError",
    "OllamaHttpTransport",
]
