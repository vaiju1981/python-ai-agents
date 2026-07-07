"""Config-driven model selection for the analytics demo.

Choose the model via environment variables so the same app runs against any
backend. Supports Ollama (local/cloud) via our adapter.
"""

from __future__ import annotations

import os

from python_ai_agents.adapters import OllamaModelPort
from python_ai_agents.core.model import ModelPort


def from_env() -> ModelPort:
    """Select a model from environment variables.

    Required: ``ANALYTICS_MODEL_PROVIDER`` (e.g. "ollama") and ``ANALYTICS_MODEL`` (e.g. "ornith:latest").
    Optional: ``OLLAMA_BASE_URL`` (default ``http://localhost:11434``).
    """
    provider = os.environ.get("ANALYTICS_MODEL_PROVIDER", "")
    model_name = os.environ.get("ANALYTICS_MODEL", "")
    if not provider or not model_name:
        raise RuntimeError(
            "set ANALYTICS_MODEL_PROVIDER (ollama) and ANALYTICS_MODEL (e.g. ornith:latest)"
        )

    if provider.lower() == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaModelPort(
            model=model_name,
            base_url=base_url,
            options={"temperature": 0},
            timeout=180.0,
        )

    raise RuntimeError(f"model provider '{provider}' is not supported (only 'ollama')")
