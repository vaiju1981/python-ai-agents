from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol
from urllib import error, request
from uuid import uuid4

import anyio

from python_ai_agents.core import AgentRequest, AgentResponse
from python_ai_agents.core.model import Message, ModelRequest, ModelResponse, Role, ToolCall, Usage

DEFAULT_OLLAMA_TEST_MODELS = (
    "gemma4:31b-cloud",
    "ornith:latest",
)


class OllamaError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


class OllamaTransport(Protocol):
    def get_json(self, base_url: str, path: str, timeout: float) -> dict[str, Any]: ...

    def post_json(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]: ...


class OllamaHttpTransport:
    """Small stdlib HTTP transport for Ollama's local API."""

    def get_json(self, base_url: str, path: str, timeout: float) -> dict[str, Any]:
        url = _join_url(base_url, path)
        http_request = request.Request(url, method="GET")
        return self._open_json(http_request, timeout)

    def post_json(
        self,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        url = _join_url(base_url, path)
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(http_request, timeout)

    def _open_json(self, http_request: request.Request, timeout: float) -> dict[str, Any]:
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                return payload
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaError(f"Ollama request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaError("Ollama request timed out") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON") from exc


@dataclass(frozen=True, slots=True)
class OllamaAgent:
    """Agent adapter for Ollama chat models."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    system_prompt: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    timeout: float = 120.0
    transport: OllamaTransport = field(default_factory=OllamaHttpTransport)

    async def run(self, request: AgentRequest) -> AgentResponse:
        payload = self._payload(request.input)
        call = partial(self.transport.post_json, self.base_url, "/api/chat", payload, self.timeout)
        data = await anyio.to_thread.run_sync(call)
        content = _chat_content(data)
        return AgentResponse.completed(content)

    async def list_models(self) -> tuple[str, ...]:
        call = partial(self.transport.get_json, self.base_url, "/api/tags", self.timeout)
        data = await anyio.to_thread.run_sync(call)
        models = data.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags response did not include a model list")
        names = [model.get("name") for model in models if isinstance(model, dict)]
        return tuple(name for name in names if isinstance(name, str))

    async def has_model(self, model: str | None = None) -> bool:
        return (model or self.model) in await self.list_models()

    def _payload(self, input_text: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": input_text})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.options:
            payload["options"] = dict(self.options)
        return payload


@dataclass(frozen=True, slots=True)
class OllamaModelPort:
    """ModelPort adapter for Ollama chat models."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    options: dict[str, Any] = field(default_factory=dict)
    timeout: float = 120.0
    transport: OllamaTransport = field(default_factory=OllamaHttpTransport)

    async def chat(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [_to_ollama_message(message) for message in request.messages],
            "stream": False,
        }
        if request.tools:
            payload["tools"] = [_to_ollama_tool(spec) for spec in request.tools]
        if self.options:
            payload["options"] = dict(self.options)
        call = partial(self.transport.post_json, self.base_url, "/api/chat", payload, self.timeout)
        data = await anyio.to_thread.run_sync(call)
        return _model_response(data)

    async def list_models(self) -> tuple[str, ...]:
        call = partial(self.transport.get_json, self.base_url, "/api/tags", self.timeout)
        data = await anyio.to_thread.run_sync(call)
        models = data.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags response did not include a model list")
        names = [model.get("name") for model in models if isinstance(model, dict)]
        return tuple(name for name in names if isinstance(name, str))

    async def has_model(self, model: str | None = None) -> bool:
        return (model or self.model) in await self.list_models()


def _chat_content(data: dict[str, Any]) -> str:
    message = data.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama chat response did not include a message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise OllamaError("Ollama chat response did not include string content")
    return content


def _model_response(data: dict[str, Any]) -> ModelResponse:
    message = data.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama chat response did not include a message object")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise OllamaError("Ollama chat response did not include string content")

    raw_tool_calls = message.get("tool_calls", [])
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise OllamaError("Ollama chat response included invalid tool calls")
    usage = Usage(
        input_tokens=_optional_int(data.get("prompt_eval_count")),
        output_tokens=_optional_int(data.get("eval_count")),
    )
    return ModelResponse(
        text=content,
        tool_calls=tuple(_to_tool_call(call) for call in raw_tool_calls),
        usage=usage,
    )


def _to_ollama_message(message: Message) -> dict[str, Any]:
    if message.role == Role.TOOL:
        return {"role": "tool", "content": message.content, "tool_name": message.tool_name}
    value: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        value["tool_calls"] = [
            {"id": call.id, "function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return value


def _to_ollama_tool(spec: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _to_tool_call(value: Any) -> ToolCall:
    if not isinstance(value, dict):
        raise OllamaError("Ollama tool call was not an object")
    function = value.get("function", {})
    if not isinstance(function, dict):
        raise OllamaError("Ollama tool call function was not an object")
    name = function.get("name")
    if not isinstance(name, str):
        raise OllamaError("Ollama tool call function did not include a name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise OllamaError("Ollama tool call arguments were not an object")
    call_id = value.get("id")
    return ToolCall(
        name=name, arguments=arguments, id=call_id if isinstance(call_id, str) else str(uuid4())
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")
