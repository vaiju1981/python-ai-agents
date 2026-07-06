"""Tests for RAG, skills, and supervision patterns."""

from __future__ import annotations

import anyio

from python_ai_agents import (
    AgentRequest,
    AgentResponse,
    DefaultAgent,
    Document,
    InMemoryVectorStore,
    Ingestor,
    KeywordBlocklistGuardrail,
    KeywordSkillSelector,
    ModelRequest,
    ModelResponse,
    RetrievalAugmentedAgent,
    RequestContext,
    RoundRobinSelector,
    SimpleSkill,
    SkillCatalog,
    SkillfulAgent,
    SupervisorAgent,
    HandoffAgent,
    GroupChatAgent,
    Turn,
)
from python_ai_agents.core.supervise import KeywordRouter


class ScriptedModel:
    def __init__(self, text: str = "ok") -> None:
        self.text = text

    async def chat(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse.text_response(self.text)


class EchoAgent:
    async def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse.completed(f"echo: {request.input[:50]}")


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Trivial embedder: bag-of-words into fixed-size vector."""
    _VOCAB = {"the": 0, "sky": 1, "is": 2, "blue": 3, "grass": 4, "green": 5}
    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * 10
        for word in text.lower().split():
            idx = self._VOCAB.get(word)
            if idx is not None:
                vec[idx] += 1.0
        return vec


def test_inmemory_vector_store_search() -> None:
    async def run() -> None:
        store = InMemoryVectorStore()
        embedder = StubEmbedder()
        ingestor = Ingestor(embedder=embedder, store=store, chunk_size=1000, chunk_overlap=0)
        await ingestor.ingest("default", [
            Document(text="The sky is blue"),
            Document(text="Grass is green"),
        ])
        results = await store.search("default", await embedder.embed("sky"), 1)
        assert len(results) >= 1
        assert "sky" in results[0].text or "blue" in results[0].text

    anyio.run(run)


def test_retrieval_augmented_agent_prepends_context() -> None:
    async def run() -> None:
        class StubRetriever:
            async def retrieve(self, tenant: str, query: str, limit: int):
                from python_ai_agents import RetrievedChunk
                return [RetrievedChunk(text="retrieved fact", score=1.0)]

        class RecordingAgent:
            def __init__(self):
                self.last_input = ""
            async def run(self, request: AgentRequest) -> AgentResponse:
                self.last_input = request.input
                return AgentResponse.completed("answer")

        recorder = RecordingAgent()
        agent = RetrievalAugmentedAgent(delegate=recorder, retriever=StubRetriever(), top_k=2)
        await agent.run(AgentRequest.ephemeral("what is X?"))
        assert "retrieved fact" in recorder.last_input
        assert "what is X?" in recorder.last_input

    anyio.run(run)


def test_retrieval_augmented_agent_delegates_when_no_results() -> None:
    async def run() -> None:
        class EmptyRetriever:
            async def retrieve(self, tenant: str, query: str, limit: int):
                return []

        agent = RetrievalAugmentedAgent(delegate=EchoAgent(), retriever=EmptyRetriever())
        response = await agent.run(AgentRequest.ephemeral("hello"))
        assert "echo: hello" in response.output

    anyio.run(run)


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def test_skill_catalog_add_and_find() -> None:
    cat = SkillCatalog()
    skill = SimpleSkill(name="math", description="math operations", instructions="Do math carefully")
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
