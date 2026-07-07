"""Tests for deterministic chart selection (pure — no data deps)."""

from __future__ import annotations

from demos.analytics.src.analytics.charts import ChartSpec, choose_chart


def test_time_series_becomes_line() -> None:
    rows = [{"period": "2024-01-01", "revenue": 10}, {"period": "2024-02-01", "revenue": 14}]
    assert choose_chart(rows) == ChartSpec("line", x="period", y="revenue", title="")


def test_category_becomes_bar() -> None:
    rows = [{"region": "west", "sales": 100}, {"region": "east", "sales": 80}]
    spec = choose_chart(rows)
    assert spec is not None and spec.kind == "bar" and spec.x == "region" and spec.y == "sales"


def test_single_numeric_becomes_histogram() -> None:
    rows = [{"amount": float(i)} for i in range(20)]
    spec = choose_chart(rows)
    assert spec is not None and spec.kind == "histogram" and spec.x == "amount"


def test_no_numeric_returns_none() -> None:
    assert choose_chart([{"a": "x", "b": "y"}]) is None


def test_empty_or_none_returns_none() -> None:
    assert choose_chart([]) is None
    assert choose_chart(None) is None


def test_booleans_are_not_measures() -> None:
    rows = [{"flag": True, "n": 1}, {"flag": False, "n": 2}]
    spec = choose_chart(rows)
    assert spec is not None and spec.y == "n"  # flag is categorical, n is the measure


def test_too_many_bars_declines() -> None:
    rows = [{"cat": f"c{i}", "v": i} for i in range(40)]
    assert choose_chart(rows, max_bars=30) is None
