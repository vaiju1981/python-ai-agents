"""Structured-output helper using Pydantic.

Calls a ``ModelPort`` and parses the response into a Pydantic ``BaseModel``,
retrying on validation failure by feeding the error back to the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.model import Message, ModelPort, ModelRequest

__all__ = [
    "StructuredOutputError",
    "extract_structured",
]

T = TypeVar("T", bound=BaseModel)

_SCHEMA_INSTRUCTION = (
    "Respond with valid JSON that matches this JSON schema. "
    "Do not include any text outside the JSON object.\n\nSchema:\n{schema}"
)


class StructuredOutputError(Exception):
    """Raised when the model output cannot be parsed after all retries."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


@dataclass(frozen=True, slots=True)
class StructuredResult:
    """Holds the parsed model and the raw model text."""

    value: BaseModel | None
    raw_text: str
    attempts: int


async def extract_structured(
    model: ModelPort,
    output_type: type[T],
    prompt: str,
    context: RequestContext,
    *,
    max_retries: int = 2,
    system_prompt: str | None = None,
) -> StructuredResult:
    """Call ``model`` and parse the response into ``output_type``.

    Builds a prompt that includes the Pydantic model's JSON schema, then
    retries up to ``max_retries`` times on validation failure, feeding the
    validation error back into the conversation.
    """
    schema = output_type.model_json_schema()
    instruction = _SCHEMA_INSTRUCTION.format(schema=json.dumps(schema, indent=2))

    messages: list[Message] = []
    if system_prompt:
        messages.append(Message.system(system_prompt))
    messages.append(Message.system(instruction))
    messages.append(Message.user(prompt))

    last_text = ""
    for attempt in range(max_retries + 1):
        response = await model.chat(ModelRequest(messages=tuple(messages)))
        last_text = response.text
        candidate = _extract_json(last_text)
        try:
            parsed = output_type.model_validate_json(candidate)
            return StructuredResult(value=parsed, raw_text=last_text, attempts=attempt + 1)
        except ValidationError as exc:
            if attempt < max_retries:
                messages.append(Message.assistant(last_text))
                messages.append(
                    Message.user(
                        f"The previous response was not valid JSON for the schema. "
                        f"Errors:\n{exc}\n\nPlease respond with corrected JSON only."
                    )
                )
            else:
                return StructuredResult(value=None, raw_text=last_text, attempts=attempt + 1)

    return StructuredResult(value=None, raw_text=last_text, attempts=max_retries + 1)


def _extract_json(text: str) -> str:
    """Extract the first JSON object from text, handling ```json fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if len(lines) >= 2:
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
    return stripped
