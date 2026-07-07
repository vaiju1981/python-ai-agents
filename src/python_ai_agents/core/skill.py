"""Skills: packaged capabilities with progressive disclosure.

A ``Skill`` has discovery metadata (name, description), detailed instructions
loaded only when selected, and optional tools. ``SkillfulAgent`` selects
relevant skills per request and builds a ``DefaultAgent`` with those skills'
instructions and tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from python_ai_agents.core.agent import AgentRequest, AgentResponse
from python_ai_agents.core.default_agent import DefaultAgent
from python_ai_agents.core.guardrail import Guardrail
from python_ai_agents.core.model import ModelPort
from python_ai_agents.core.observe import AgentObserver
from python_ai_agents.core.tool import DenyEffectfulTools, Tool, ToolApprover
from python_ai_agents.core.trust import Trust

__all__ = [
    "KeywordSkillSelector",
    "Skill",
    "SkillCatalog",
    "SkillSelector",
    "SkillfulAgent",
    "SimpleSkill",
]


@dataclass(frozen=True, slots=True)
class SimpleSkill:
    """A simple skill with name, description, instructions, and optional tools."""

    name: str
    description: str
    instructions: str
    tools: tuple[Tool, ...] = ()


class Skill(Protocol):
    """A packaged capability (Anthropic-style).

    Discovery metadata (``name``, ``description``) is always visible; detailed
    ``instructions`` are loaded only when the skill is selected.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def instructions(self) -> str: ...

    @property
    def tools(self) -> tuple[Tool, ...]: ...


class SkillCatalog:
    """A registry of skills."""

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: list[Skill] = list(skills or [])

    def add(self, skill: Skill) -> None:
        self._skills.append(skill)

    def all(self) -> list[Skill]:
        return list(self._skills)

    def find(self, name: str) -> Skill | None:
        for s in self._skills:
            if s.name == name:
                return s
        return None


class SkillSelector(Protocol):
    """Selects relevant skills for a given input."""

    def select(self, catalog: SkillCatalog, input_text: str) -> list[Skill]: ...


@dataclass(frozen=True, slots=True)
class KeywordSkillSelector:
    """Selects skills whose name or description contains any of the keywords."""

    keywords: set[str] = field(default_factory=set)

    def select(self, catalog: SkillCatalog, input_text: str) -> list[Skill]:
        if not self.keywords:
            return catalog.all()
        text_lower = input_text.lower()
        selected = []
        for s in catalog.all():
            name_desc = (s.name + " " + s.description).lower()
            # A skill is selected if any keyword appears in BOTH
            # the skill's name/description AND the input text.
            if any(kw.lower() in name_desc and kw.lower() in text_lower for kw in self.keywords):
                selected.append(s)
        return selected


@dataclass(slots=True)
class SkillfulAgent:
    """An agent that equips itself per request with selected skills.

    Selects relevant skills, then builds a ``DefaultAgent`` with those skills'
    instructions appended to the system prompt and their tools registered.
    Progressive disclosure — only the selected skills' instructions enter the
    model's context.
    """

    model: ModelPort
    base_prompt: str = ""
    registry: SkillCatalog = field(default_factory=SkillCatalog)
    selector: SkillSelector = field(default_factory=KeywordSkillSelector)
    base_tools: list[Tool] = field(default_factory=list)
    guardrails: list[Guardrail] = field(default_factory=list)
    observers: list[AgentObserver] = field(default_factory=list)
    tool_approver: ToolApprover = field(default_factory=DenyEffectfulTools)
    max_steps: int = 8

    async def run(self, request: AgentRequest) -> AgentResponse:
        selected = self.selector.select(self.registry, request.input)
        prompt_parts: list[str] = []
        if self.base_prompt:
            prompt_parts.append(self.base_prompt)
        tools: list[Tool] = list(self.base_tools)
        for skill in selected:
            prompt_parts.append(f"# Skill: {skill.name}\n{skill.instructions}")
            tools.extend(skill.tools)

        agent = DefaultAgent(
            model=self.model,
            tools=tools,
            system_prompt="\n\n".join(prompt_parts) if prompt_parts else None,
            max_steps=self.max_steps,
            observers=list(self.observers),
            tool_approver=self.tool_approver,
        )
        if not self.guardrails:
            return await agent.run(request)
        return await Trust.govern(agent, guardrails=list(self.guardrails)).run(request)
