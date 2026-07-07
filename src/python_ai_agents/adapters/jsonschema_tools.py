from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from python_ai_agents.core.context import RequestContext
from python_ai_agents.core.tool import ToolDecision, ToolSpec


class JsonSchemaToolArgumentValidator:
    """Tool argument validator backed by the jsonschema package."""

    async def validate(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        context: RequestContext,
    ) -> ToolDecision:
        try:
            Draft202012Validator.check_schema(spec.input_schema)
            Draft202012Validator(spec.input_schema).validate(arguments)
        except ValidationError as exc:
            return ToolDecision.deny(
                f"tool '{spec.name}' arguments failed schema validation: {exc.message}"
            )
        return ToolDecision.allow()
