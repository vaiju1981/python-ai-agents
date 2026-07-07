from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse


class IdempotencyStore(Protocol):
    """Stores prior turn results by tenant and caller-scoped idempotency key."""

    async def lookup(self, tenant: str, key: str) -> AgentResponse | None: ...

    async def save(self, tenant: str, key: str, response: AgentResponse) -> None: ...


class InMemoryIdempotencyStore:
    """Bounded in-memory idempotency store. First write wins."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max(1, max_entries)
        self._entries: OrderedDict[tuple[str, str], AgentResponse] = OrderedDict()

    async def lookup(self, tenant: str, key: str) -> AgentResponse | None:
        return self._entries.get((tenant, key))

    async def save(self, tenant: str, key: str, response: AgentResponse) -> None:
        scoped_key = (tenant, key)
        if scoped_key not in self._entries:
            self._entries[scoped_key] = response
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


@dataclass(frozen=True, slots=True)
class IdempotentAgent:
    """Returns a prior result when a request repeats the same idempotency key."""

    delegate: Agent
    store: IdempotencyStore
    key_attribute: str = "idempotencyKey"

    async def run(self, request: AgentRequest) -> AgentResponse:
        key = request.context.attributes.get(self.key_attribute)
        if key is None or not key.strip():
            return await self.delegate.run(request)

        scoped_key = _scope_key(
            principal=request.context.principal,
            session_id=request.context.session_id,
            key=key,
        )
        prior = await self.store.lookup(request.context.tenant, scoped_key)
        if prior is not None:
            return prior

        response = await self.delegate.run(request)
        if not response.retryable:
            await self.store.save(request.context.tenant, scoped_key, response)
        return response


def _scope_key(*, principal: str, session_id: str, key: str) -> str:
    return f"{principal}\0{session_id}\0{key}"
