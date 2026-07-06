"""Tests for tokenizer, prompt template, and pricing utilities."""

from __future__ import annotations

import pytest

from python_ai_agents import (
    HeuristicTokenizer,
    Message,
    Pricing,
    PromptTemplate,
    Role,
    TokenPrice,
    TokenWindowedMemory,
    TokenWindowedMemory,
    Usage,
)
from python_ai_agents.core.memory import SummarizingMemory


def test_heuristic_tokenizer_estimates() -> None:
    tok = HeuristicTokenizer()
    assert tok.count_tokens("") == 0
    assert tok.count_tokens("ab") == 1  # 2 chars → 1 token
    assert tok.count_tokens("abcdefgh") == 2  # 8 chars → 2 tokens


def test_token_windowed_memory_keeps_within_budget() -> None:
    tok = HeuristicTokenizer()
    mem = TokenWindowedMemory(tokenizer=tok, max_tokens=10)
    mem.add(Message.system("sys"))  # 3 chars → 1 token
    for i in range(10):
        mem.add(Message.user(f"msg{i}"))  # 5 chars → 2 tokens each
    history = mem.history()
    assert history[0].role == Role.SYSTEM
    # Should have kept only recent messages that fit
    assert len(history) >= 2  # system + at least one recent


def test_prompt_template_render() -> None:
    template = PromptTemplate("Hello {name}, you are {role}.")
    assert set(template.variables) == {"name", "role"}
    result = template.render({"name": "Alice", "role": "admin"})
    assert result == "Hello Alice, you are admin."


def test_prompt_template_missing_value_raises() -> None:
    template = PromptTemplate("Hello {name}")
    with pytest.raises(KeyError):
        template.render({})


def test_prompt_template_no_placeholders() -> None:
    template = PromptTemplate("plain text")
    assert template.variables == []
    assert template.render({}) == "plain text"


def test_token_price_cost() -> None:
    price = TokenPrice(input_per_1m=2.5, output_per_1m=10.0)
    cost = price.cost(Usage(input_tokens=1_000_000, output_tokens=500_000))
    assert cost == 2.5 + 5.0  # 7.5


def test_token_price_free() -> None:
    assert TokenPrice(0, 0).cost(Usage(input_tokens=100, output_tokens=100)) == 0.0


def test_pricing_cost_by_model() -> None:
    pricing = Pricing(prices={
        "gpt-4o": TokenPrice(input_per_1m=2.5, output_per_1m=10.0),
    })
    cost = pricing.cost("gpt-4o", Usage(input_tokens=1_000_000, output_tokens=0))
    assert cost == 2.5
    # Unknown model → free
    assert pricing.cost("unknown", Usage(input_tokens=1000, output_tokens=1000)) == 0.0


def test_pricing_total_across_models() -> None:
    pricing = Pricing(prices={
        "gpt-4o": TokenPrice(input_per_1m=2.5, output_per_1m=10.0),
        "claude": TokenPrice(input_per_1m=3.0, output_per_1m=15.0),
    })
    total = pricing.total({
        "gpt-4o": Usage(input_tokens=1_000_000, output_tokens=0),
        "claude": Usage(input_tokens=0, output_tokens=1_000_000),
    })
    assert total == 2.5 + 15.0


def test_summarizing_memory_folds_old_messages() -> None:
    tok = HeuristicTokenizer()
    calls = []

    class StubSummarizer:
        def summarize(self, messages):
            texts = " ".join(m.content for m in messages)
            calls.append(texts)
            return f"summary[{texts[:20]}]"

    mem = SummarizingMemory(
        tokenizer=tok, summarizer=StubSummarizer(),
        max_tokens=5, min_recent=2,
    )
    mem.add(Message.system("sys"))
    for i in range(5):
        mem.add(Message.user(f"message number {i}"))
    history = mem.history()
    # Should have system + summary + recent messages
    # System messages include the original "sys" and possibly a summary
    summaries = [m for m in history if m.role == Role.SYSTEM and m.content != "sys"]
    assert any("summary" in m.content.lower() for m in summaries)
    assert len(calls) > 0  # summarizer was called
