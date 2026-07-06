import os
from typing import Any

import anyio
import pytest

from python_ai_agents import AgentRequest
from python_ai_agents.adapters import DEFAULT_OLLAMA_TEST_MODELS, OllamaAgent


class FakeOllamaTransport:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def get_json(self, base_url: str, path: str, timeout: float) -> dict[str, Any]:
        return {
            "models": [
                {"name": "ornith:latest"},
                {"name": "hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0"},
            ]
        }

    def post_json(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        self.payloads.append(payload)
        return {"message": {"role": "assistant", "content": "pong"}}


def test_ollama_agent_builds_non_streaming_chat_payload() -> None:
    async def run() -> None:
        transport = FakeOllamaTransport()
        agent = OllamaAgent(
            "ornith:latest",
            system_prompt="Answer tersely.",
            options={"temperature": 0},
            transport=transport,
        )

        response = await agent.run(AgentRequest.ephemeral("ping"))

        assert response.output == "pong"
        assert transport.payloads == [
            {
                "model": "ornith:latest",
                "messages": [
                    {"role": "system", "content": "Answer tersely."},
                    {"role": "user", "content": "ping"},
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
        ]

    anyio.run(run)


def test_ollama_agent_lists_models() -> None:
    async def run() -> None:
        agent = OllamaAgent("ornith:latest", transport=FakeOllamaTransport())

        assert await agent.has_model()
        assert await agent.has_model("hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0")
        assert not await agent.has_model("gemma4:31b-cloud")

    anyio.run(run)


@pytest.mark.skipif(
    os.environ.get("PAA_RUN_OLLAMA_TESTS") != "1",
    reason="set PAA_RUN_OLLAMA_TESTS=1 to run live Ollama model smoke tests",
)
@pytest.mark.ollama
@pytest.mark.parametrize("model", DEFAULT_OLLAMA_TEST_MODELS)
def test_live_ollama_models_respond(model: str) -> None:
    async def run() -> None:
        agent = OllamaAgent(model, options={"temperature": 0}, timeout=180)
        if not await agent.has_model():
            pytest.skip(f"Ollama model is not available: {model}")

        response = await agent.run(AgentRequest.ephemeral("Reply with exactly: ok"))

        assert response.output.strip()

    anyio.run(run)
