"""Tests for token pricing utilities."""

from __future__ import annotations

from python_ai_agents import Pricing, TokenPrice, Usage


def test_token_price_cost() -> None:
    price = TokenPrice(input_per_1m=2.5, output_per_1m=10.0)
    cost = price.cost(Usage(input_tokens=1_000_000, output_tokens=500_000))
    assert cost == 2.5 + 5.0  # 7.5


def test_token_price_free() -> None:
    assert TokenPrice(0, 0).cost(Usage(input_tokens=100, output_tokens=100)) == 0.0


def test_pricing_cost_by_model() -> None:
    pricing = Pricing(
        prices={
            "gpt-4o": TokenPrice(input_per_1m=2.5, output_per_1m=10.0),
        }
    )
    cost = pricing.cost("gpt-4o", Usage(input_tokens=1_000_000, output_tokens=0))
    assert cost == 2.5
    # Unknown model → free
    assert pricing.cost("unknown", Usage(input_tokens=1000, output_tokens=1000)) == 0.0


def test_pricing_total_across_models() -> None:
    pricing = Pricing(
        prices={
            "gpt-4o": TokenPrice(input_per_1m=2.5, output_per_1m=10.0),
            "claude": TokenPrice(input_per_1m=3.0, output_per_1m=15.0),
        }
    )
    total = pricing.total(
        {
            "gpt-4o": Usage(input_tokens=1_000_000, output_tokens=0),
            "claude": Usage(input_tokens=0, output_tokens=1_000_000),
        }
    )
    assert total == 2.5 + 15.0
