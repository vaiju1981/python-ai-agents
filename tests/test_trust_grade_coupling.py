"""PR-6 verification: LLM trust-grade coupling test.

The linchpin of the "defensible answers" story is that the graded trust tier is
not merely prose buried in the answer body, but a *structured* ``ToolResult.trust``
field that the tool-result formatter renders as a machine-checkable
``[TRUST:TIER]`` token the model is instructed to honor (abstain from causal
claims on ``[TRUST:INSUFFICIENT]``).

Gates from docs/production_readiness.md:
1. Format a ``ModelsToolset._ok`` result at each tier -> the emitted token matches
   the expected string per tier.
2. Feed an ``INSUFFICIENT`` causal-style result into the agent formatter -> the
   rendered message instructs non-assertion, and the agent system prompt tells
   the model to honor it.
3. (Optional, flagged) run the agent on thin-evidence data and assert the final
   answer abstains rather than asserting a confident causal claim.
"""

from __future__ import annotations

import os

import pytest

from demos.analytics.src.analytics.agent import create_agent
from demos.analytics.src.analytics.csv_source import CsvSource
from demos.analytics.src.analytics.models_tools import ModelsToolset
from demos.analytics.src.analytics.profiler import profile_dataset
from demos.analytics.src.analytics.semantic_model import SemanticModel
from demos.analytics.src.analytics.toolset import AnalyticsToolset
from python_ai_agents.core.default_agent import _tool_result_for_model


def _toolset(tmp_path):
    csv = tmp_path / "sales.csv"
    csv.write_text("region,amount\nNorth,1.0\nSouth,2.0\n")
    src = CsvSource(named_csvs={"sales": csv})
    model = SemanticModel.from_profile(profile_dataset(src))
    return src, ModelsToolset(src, model), AnalyticsToolset(src, model)


@pytest.mark.parametrize("tier", ["TRUSTED", "DIRECTIONAL", "INSUFFICIENT"])
def test_trust_token_emitted_per_tier(tmp_path, tier):
    src, models, _ = _toolset(tmp_path)
    trust = {
        "tier": tier,
        "confidence": 0.5,
        "reasons": [],
        "abstain": tier == "INSUFFICIENT",
    }
    result = models._ok("build_model", {"n": 250}, trust=trust)
    assert result.trust == trust
    rendered = _tool_result_for_model("build_model", result, frame=True)
    assert f"[TRUST:{tier}]" in rendered
    if tier == "INSUFFICIENT":
        assert "do not assert causal" in rendered
    src.close()


def test_agent_prompt_instructs_abstention_on_insufficient(tmp_path):
    src, _models, analytics = _toolset(tmp_path)
    # A causal-style result graded INSUFFICIENT (e.g. thin-evidence matched_impact).
    trust = {
        "tier": "INSUFFICIENT",
        "confidence": 0.0,
        "reasons": ["n=4 below abstain threshold"],
        "abstain": True,
    }
    result = analytics._ok("matched_impact result", trust=trust)
    rendered = _tool_result_for_model("matched_impact", result, frame=True)
    # The formatted message carries the machine-checkable marker + non-assertion instruction.
    assert "[TRUST:INSUFFICIENT]" in rendered
    assert "do not assert causal" in rendered

    # The agent's system prompt must instruct the model to honor the marker.
    agent = create_agent(src, object(), analytics.model)
    assert "[TRUST:INSUFFICIENT]" in agent.system_prompt
    assert "do NOT make causal or confident" in agent.system_prompt
    src.close()


@pytest.mark.skipif(
    not os.getenv("PAA_RUN_OLLAMA_TESTS"),
    reason="live LLM check; set PAA_RUN_OLLAMA_TESTS=1 with a running Ollama",
)
def test_live_agent_abstains_on_thin_evidence(tmp_path):
    import anyio

    from demos.analytics.src.analytics.models import from_env as model_from_env
    from python_ai_agents import AgentRequest
    from python_ai_agents.core.context import RequestContext

    # Tiny, sparse panel so matched_impact grades INSUFFICIENT.
    csv = tmp_path / "panel.csv"
    csv.write_text(
        "asset,day,coin\n"
        "A,2024-01-01,10\n"
        "A,2024-01-10,20\n"
        "B,2024-01-01,30\n"
        "B,2024-01-10,40\n"
    )
    events = tmp_path / "events.csv"
    events.write_text("asset,day\nA,2024-01-05\n")
    src = CsvSource(named_csvs={"metric": csv, "events": events})
    model = SemanticModel.from_profile(profile_dataset(src))
    agent = create_agent(src, model_from_env(), model)

    async def run():
        return await agent.run(
            AgentRequest(
                "What is the causal impact of the treatment on coin?",
                RequestContext.ephemeral(),
            )
        )

    response = anyio.run(run)
    low = response.output.lower()
    assert "insufficient" in low or "not enough" in low or "cannot" in low
    src.close()
