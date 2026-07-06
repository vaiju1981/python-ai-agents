"""Tests for the Pydantic structured-output helper."""

from __future__ import annotations

import anyio
from pydantic import BaseModel

from python_ai_agents import (
    ModelRequest,
    ModelResponse,
    RequestContext,
    Usage,
    extract_structured,
)


class PersonInfo(BaseModel):
    name: str
    age: int
    email: str | None = None


class ScriptedModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def chat(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.responses:
            text = self.responses.pop(0)
        else:
            text = '{"name": "fallback", "age": 0}'
        return ModelResponse.text_response(text, Usage(input_tokens=10, output_tokens=10))


def test_extract_structured_parses_valid_json() -> None:
    async def run() -> None:
        model = ScriptedModel(['{"name": "Alice", "age": 30, "email": "alice@test.com"}'])
        result = await extract_structured(
            model, PersonInfo, "Extract person info from: Alice is 30 years old",
            RequestContext.ephemeral(),
        )

        assert result.value is not None
        assert result.value.name == "Alice"
        assert result.value.age == 30
        assert result.value.email == "alice@test.com"
        assert result.attempts == 1

    anyio.run(run)


def test_extract_structured_handles_json_fences() -> None:
    async def run() -> None:
        model = ScriptedModel(['```json\n{"name": "Bob", "age": 25}\n```'])
        result = await extract_structured(
            model, PersonInfo, "Bob is 25",
            RequestContext.ephemeral(),
        )

        assert result.value is not None
        assert result.value.name == "Bob"
        assert result.value.age == 25

    anyio.run(run)


def test_extract_structured_retries_on_invalid_json() -> None:
    async def run() -> None:
        model = ScriptedModel([
            "not json at all",
            '{"name": "Charlie", "age": 40}',
        ])
        result = await extract_structured(
            model, PersonInfo, "Charlie is 40",
            RequestContext.ephemeral(),
            max_retries=2,
        )

        assert result.value is not None
        assert result.value.name == "Charlie"
        assert result.attempts == 2

    anyio.run(run)


def test_extract_structured_returns_none_after_max_retries() -> None:
    async def run() -> None:
        model = ScriptedModel(["invalid", "still invalid", "nope"])
        result = await extract_structured(
            model, PersonInfo, "test",
            RequestContext.ephemeral(),
            max_retries=2,
        )

        assert result.value is None
        assert result.attempts == 3
        assert result.raw_text == "nope"

    anyio.run(run)


def test_extract_structured_includes_schema_in_prompt() -> None:
    async def run() -> None:
        model = ScriptedModel(['{"name": "Test", "age": 1}'])
        await extract_structured(
            model, PersonInfo, "test input",
            RequestContext.ephemeral(),
        )

        # The model should have received a system message with schema instructions
        first_request = model.requests[0]
        system_messages = [m for m in first_request.messages if m.role.value == "system"]
        assert len(system_messages) >= 1
        assert "JSON" in system_messages[0].content
        assert "name" in system_messages[0].content

    anyio.run(run)


def test_extract_structured_with_system_prompt() -> None:
    async def run() -> None:
        model = ScriptedModel(['{"name": "Dan", "age": 50}'])
        result = await extract_structured(
            model, PersonInfo, "Dan is 50",
            RequestContext.ephemeral(),
            system_prompt="You are a helpful assistant.",
        )

        assert result.value is not None
        assert result.value.name == "Dan"

        # Check system prompt was included
        first_request = model.requests[0]
        system_contents = [m.content for m in first_request.messages if m.role.value == "system"]
        assert any("helpful assistant" in c for c in system_contents)

    anyio.run(run)
