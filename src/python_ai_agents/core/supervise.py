"""Supervision patterns: supervisor, handoff, and group-chat agents.

These are thin orchestration seams over the ``Agent`` protocol. For more
complex multi-agent workflows, use the LangGraph adapter
(``adapters.langgraph``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from python_ai_agents.core.agent import Agent, AgentRequest, AgentResponse

__all__ = [
    "GroupChatAgent",
    "Handoff",
    "HandoffAgent",
    "KeywordRouter",
    "Manager",
    "ManagerAgent",
    "Router",
    "RoundRobinSelector",
    "SpeakerSelector",
    "SupervisorAgent",
]


# ---------------------------------------------------------------------------
# Router + Supervisor
# ---------------------------------------------------------------------------


class Router(Protocol):
    """Picks the best specialist by name for a given input."""

    def route(self, input_text: str, descriptions: dict[str, str]) -> str: ...


@dataclass(frozen=True, slots=True)
class KeywordRouter:
    """Routes to the specialist whose description contains a keyword from the input."""

    def route(self, input_text: str, descriptions: dict[str, str]) -> str:
        text_lower = input_text.lower()
        for name, desc in descriptions.items():
            if any(word in desc.lower() for word in text_lower.split()):
                return name
        return next(iter(descriptions), "")


@dataclass(slots=True)
class SupervisorAgent:
    """Routes each request to one of several named specialist agents.

    A ``Router`` picks the best specialist; if it returns an unknown name, the
    request goes to the fallback (the first registered specialist). Itself an
    ``Agent``, so it composes.
    """

    agents: dict[str, Agent]
    descriptions: dict[str, str]
    router: Router
    fallback: str = ""

    async def run(self, request: AgentRequest) -> AgentResponse:
        chosen = self.router.route(request.input, self.descriptions)
        agent = self.agents.get(chosen) or self.agents.get(self.fallback)
        if agent is None:
            agent = next(iter(self.agents.values()))
        return await agent.run(request)

    @staticmethod
    def builder() -> _SupervisorBuilder:
        return _SupervisorBuilder()


@dataclass
class _SupervisorBuilder:
    _agents: dict[str, Agent] = field(default_factory=dict)
    _descriptions: dict[str, str] = field(default_factory=dict)
    _router: Router | None = None
    _fallback: str = ""
    _first: str = ""

    def specialist(self, name: str, description: str, agent: Agent) -> _SupervisorBuilder:
        if not self._first:
            self._first = name
        self._agents[name] = agent
        self._descriptions[name] = description
        return self

    def router(self, router: Router) -> _SupervisorBuilder:
        self._router = router
        return self

    def fallback(self, name: str) -> _SupervisorBuilder:
        self._fallback = name
        return self

    def build(self) -> SupervisorAgent:
        if not self._agents:
            raise ValueError("at least one specialist is required")
        return SupervisorAgent(
            agents=dict(self._agents),
            descriptions=dict(self._descriptions),
            router=self._router or KeywordRouter(),
            fallback=self._fallback or self._first,
        )


# ---------------------------------------------------------------------------
# Handoff (Swarm pattern)
# ---------------------------------------------------------------------------


class Handoff(Protocol):
    """Decides whether a peer agent should take over after each hop."""

    def handoff(self, output: str, descriptions: dict[str, str]) -> str | None: ...


@dataclass(slots=True)
class HandoffAgent:
    """Routes through peer agents that can hand off control to one another.

    A starting agent handles the request; after each hop a ``Handoff`` decides
    whether a peer should take over. Control moves laterally between peers
    until one keeps it or the hop budget is spent.
    """

    agents: dict[str, Agent]
    descriptions: dict[str, str]
    start: str
    handoff: Handoff
    max_hops: int = 5

    async def run(self, request: AgentRequest) -> AgentResponse:
        ctx = request.context
        task = request.input
        current = self.start
        response: AgentResponse | None = None

        for _hop in range(1, self.max_hops + 1):
            response = await self.agents[current].run(AgentRequest(input=task, context=ctx))
            if response.blocked:
                return response
            next_agent = self.handoff.handoff(response.output, self.descriptions)
            if next_agent is None or next_agent == current:
                return response
            if next_agent not in self.agents:
                return response
            current = next_agent
            task = response.output

        return response  # type: ignore[return-value]

    @staticmethod
    def builder() -> _HandoffBuilder:
        return _HandoffBuilder()


@dataclass
class _HandoffBuilder:
    _agents: dict[str, Agent] = field(default_factory=dict)
    _descriptions: dict[str, str] = field(default_factory=dict)
    _handoff: Handoff | None = None
    _start: str = ""
    _first: str = ""
    _max_hops: int = 5

    def agent(self, name: str, description: str, agent: Agent) -> _HandoffBuilder:
        if not self._first:
            self._first = name
        self._agents[name] = agent
        self._descriptions[name] = description
        return self

    def handoff(self, handoff: Handoff) -> _HandoffBuilder:
        self._handoff = handoff
        return self

    def start(self, name: str) -> _HandoffBuilder:
        self._start = name
        return self

    def max_hops(self, n: int) -> _HandoffBuilder:
        self._max_hops = n
        return self

    def build(self) -> HandoffAgent:
        if not self._agents:
            raise ValueError("at least one agent is required")
        if self._handoff is None:
            raise ValueError("handoff is required")
        return HandoffAgent(
            agents=dict(self._agents),
            descriptions=dict(self._descriptions),
            start=self._start or self._first,
            handoff=self._handoff,
            max_hops=self._max_hops,
        )


# ---------------------------------------------------------------------------
# Group chat (AutoGen pattern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    """One turn in a group-chat transcript."""

    speaker: str
    content: str


class SpeakerSelector(Protocol):
    """Picks who speaks next in a group chat."""

    def next(
        self, task: str, transcript: list[Turn], descriptions: dict[str, str]
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class RoundRobinSelector:
    """Cycles through speakers in registration order."""

    def next(self, task: str, transcript: list[Turn], descriptions: dict[str, str]) -> str | None:
        names = list(descriptions)
        if not names:
            return None
        speakers = [t.speaker for t in transcript if t.speaker != "user"]
        if not speakers:
            return names[0]
        last = speakers[-1]
        idx = names.index(last) if last in names else -1
        next_idx = (idx + 1) % len(names)
        # Stop after a full cycle
        if len(speakers) >= len(names):
            return None
        return names[next_idx]


class Manager(Protocol):
    """Decides which worker to delegate to and what subtask to give it."""

    def delegate(self, task: str, descriptions: dict[str, str]) -> tuple[str, str]: ...


@dataclass(slots=True)
class ManagerAgent:
    """A manager that delegates subtasks to specialist workers.

    The manager decomposes the task and delegates each subtask to a specialist.
    Itself an ``Agent``, so it composes.
    """

    agents: dict[str, Agent]
    descriptions: dict[str, str]
    manager: Manager
    max_subtasks: int = 5

    async def run(self, request: AgentRequest) -> AgentResponse:
        results: list[str] = []
        task = request.input
        for _ in range(self.max_subtasks):
            worker_name, subtask = self.manager.delegate(task, self.descriptions)
            if not worker_name or worker_name not in self.agents:
                break
            resp = await self.agents[worker_name].run(
                AgentRequest(input=subtask, context=request.context.child_session())
            )
            results.append(f"[{worker_name}]: {resp.output}")
            if resp.blocked:
                return resp
            task = resp.output
        return AgentResponse.completed("\n".join(results))


@dataclass(slots=True)
class GroupChatAgent:
    """Runs a group chat: several agents share one transcript.

    A ``SpeakerSelector`` picks who speaks next each round. Every speaker sees
    the full shared conversation and adds to it. The chat ends when the
    selector returns no next speaker or the round budget is spent.
    """

    agents: dict[str, Agent]
    descriptions: dict[str, str]
    selector: SpeakerSelector
    max_rounds: int = 5

    async def run(self, request: AgentRequest) -> AgentResponse:
        ctx = request.context
        transcript: list[Turn] = [Turn(speaker="user", content=request.input)]
        last: AgentResponse | None = None

        for _round in range(1, self.max_rounds + 1):
            speaker = self.selector.next(request.input, transcript, self.descriptions)
            if speaker is None or speaker not in self.agents:
                break
            rendered = "\n".join(f"{t.speaker}: {t.content}" for t in transcript)
            resp = await self.agents[speaker].run(
                AgentRequest(input=rendered, context=ctx.child_session())
            )
            if resp.blocked:
                return resp
            transcript.append(Turn(speaker=speaker, content=resp.output))
            last = resp

        if last is None:
            return AgentResponse.completed("")
        return last
