"""Token-usage cost accounting with a bring-your-own price table."""

from __future__ import annotations

from dataclasses import dataclass

from python_ai_agents.core.model import Usage

__all__ = ["Pricing", "TokenPrice"]


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """Per-token pricing per 1,000,000 tokens — the unit vendors quote in.

    ``FREE`` models a local/no-cost model (e.g. Ollama).
    """

    input_per_1m: float
    output_per_1m: float

    def __post_init__(self) -> None:
        if self.input_per_1m < 0 or self.output_per_1m < 0:
            raise ValueError("prices must be >= 0")

    def cost(self, usage: Usage) -> float:
        return (
            self.input_per_1m * (usage.input_tokens or 0) / 1_000_000.0
            + self.output_per_1m * (usage.output_tokens or 0) / 1_000_000.0
        )


FREE_PRICE = TokenPrice(0, 0)


@dataclass(frozen=True, slots=True)
class Pricing:
    """Turns token usage into cost given a price table keyed by model name.

    A model with no registered price is treated as ``TokenPrice.FREE``.
    """

    prices: dict[str, TokenPrice]

    def price_of(self, model: str) -> TokenPrice:
        return self.prices.get(model, FREE_PRICE)

    def cost(self, model: str, usage: Usage) -> float:
        return self.price_of(model).cost(usage)

    def total(self, usage_by_model: dict[str, Usage]) -> float:
        return sum(self.cost(model, usage) for model, usage in usage_by_model.items())
