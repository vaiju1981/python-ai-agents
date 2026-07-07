"""Config-driven model selection for the analytics demo.

Choose the model via environment variables so the same app runs against any
backend. Supports Ollama (local/cloud) via our adapter.
"""

from __future__ import annotations

import os

from python_ai_agents.adapters import RECOMMENDED_SAMPLING, OllamaModelPort
from python_ai_agents.core.model import ModelPort

MAX_OLLAMA_CONTEXT = 131_072
DEFAULT_OLLAMA_CONTEXT = 65_536


def from_env() -> ModelPort:
    """Select a model from environment variables.

    Required: ``ANALYTICS_MODEL_PROVIDER`` (e.g. "ollama") and
    ``ANALYTICS_MODEL`` (e.g. "ornith:latest").
    Optional: ``OLLAMA_BASE_URL`` (default ``http://localhost:11434``),
    ``OLLAMA_NUM_CTX`` (default ``65536``, max ``131072``), and sampling overrides
    ``OLLAMA_TEMPERATURE`` / ``OLLAMA_TOP_P`` / ``OLLAMA_TOP_K``.
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
            options=_ollama_options(),
            timeout=180.0,
        )

    raise RuntimeError(f"model provider '{provider}' is not supported (only 'ollama')")


def _ollama_options() -> dict[str, int | float]:
    # Sampling defaults follow the vendor guidance (RECOMMENDED_SAMPLING): temperature 0
    # made the local model repeat tool calls until the step budget on open-ended prompts.
    # Override per model via env.
    requested_ctx = int(os.environ.get("OLLAMA_NUM_CTX", DEFAULT_OLLAMA_CONTEXT))
    rec = RECOMMENDED_SAMPLING
    return {
        "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", rec["temperature"])),
        "top_p": float(os.environ.get("OLLAMA_TOP_P", rec["top_p"])),
        "top_k": int(os.environ.get("OLLAMA_TOP_K", rec["top_k"])),
        "num_ctx": min(requested_ctx, MAX_OLLAMA_CONTEXT),
    }
