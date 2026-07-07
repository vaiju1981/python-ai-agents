"""Tests for skills and multi-agent supervision patterns."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    GroupChatAgent,
    HandoffAgent,
    KeywordBlocklistGuardrail,
    KeywordSkillSelector,
    ModelRequest,
    ModelResponse,
    RoundRobinSelector,
    SimpleSkill,
    SkillCatalog,
    SkillfulAgent,
    SupervisorAgent,
)
from python_ai_agents.core.supervise import KeywordRouter


class ScriptedModel:
    def __init__(self, text: str = "ok") -> None:
        self.text = text

    async def chat(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse.text_response(self.text)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_skill_catalog_add_and_find() -> None:
    cat = SkillCatalog()
    skill = SimpleSkill(
        name="math", description="math operations", instructions="Do math carefully"
    )
    cat.add(skill)
    assert cat.find("math") is skill
    assert cat.find("nonexistent") is None
    assert len(cat.all()) == 1


def test_keyword_skill_selector_selects_by_keyword() -> None:
    cat = SkillCatalog()
    cat.add(SimpleSkill(name="math", description="math operations", instructions=""))
    cat.add(SimpleSkill(name="search", description="web search", instructions=""))
    selector = KeywordSkillSelector(keywords={"math"})
    selected = selector.select(cat, "math the sum")
    assert len(selected) == 1
    assert selected[0].name == "math"


def test_skillful_agent_equips_selected_skills() -> None:
    async def run() -> None:
        model = ScriptedModel("skillful answer")
        cat = SkillCatalog()
        cat.add(SimpleSkill(name="math", description="math", instructions="Be precise"))
        agent = SkillfulAgent(
            model=model,
            registry=cat,
            selector=KeywordSkillSelector(keywords={"math"}),
            base_prompt="You are helpful.",
        )
        response = await agent.run(AgentRequest.ephemeral("do math"))
        assert response.output == "skillful answer"

    anyio.run(run)


def test_skillful_agent_applies_guardrails() -> None:
    async def run() -> None:
        agent = SkillfulAgent(
            model=ScriptedModel("should not run"),
            guardrails=[KeywordBlocklistGuardrail(keywords={"blocked"})],
        )
        response = await agent.run(AgentRequest.ephemeral("blocked request"))
        assert response.blocked

    anyio.run(run)


# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------


def test_supervisor_agent_routes_to_specialist() -> None:
    async def run() -> None:
        class MathAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed("math answer")

        class CodeAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed("code answer")

        agent = (
            SupervisorAgent.builder()
            .specialist("math", "mathematics and numbers", MathAgent())
            .specialist("code", "programming and code", CodeAgent())
            .router(KeywordRouter())
            .build()
        )
        r1 = await agent.run(AgentRequest.ephemeral("calculate numbers"))
        assert "math answer" in r1.output

    anyio.run(run)


def test_handoff_agent_transfers_control() -> None:
    async def run() -> None:
        class FirstAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed("handoff to second")

        class SecondAgent:
            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed("final answer")

        class SimpleHandoff:
            def handoff(self, output: str, descriptions: dict[str, str]) -> str | None:
                if "handoff to second" in output:
                    return "second"
                return None

        agent = (
            HandoffAgent.builder()
            .agent("first", "first agent", FirstAgent())
            .agent("second", "second agent", SecondAgent())
            .start("first")
            .handoff(SimpleHandoff())
            .max_hops(3)
            .build()
        )
        response = await agent.run(AgentRequest.ephemeral("start"))
        assert "final answer" in response.output

    anyio.run(run)


def test_group_chat_agent_rounds() -> None:
    async def run() -> None:
        class SimpleAgent:
            def __init__(self, name: str):
                self.name = name

            async def run(self, request: AgentRequest) -> AgentResponse:
                return AgentResponse.completed(f"{self.name} speaks")

        agent = GroupChatAgent(
            agents={"a": SimpleAgent("A"), "b": SimpleAgent("B")},
            descriptions={"a": "agent A", "b": "agent B"},
            selector=RoundRobinSelector(),
            max_rounds=2,
        )
        response = await agent.run(AgentRequest.ephemeral("discuss"))
        assert "speaks" in response.output

    anyio.run(run)
