"""Chart selection: pick a sensible chart for a result set (deterministic).

Pure logic — ``choose_chart`` returns a ``ChartSpec`` describing *what* to plot;
the UI turns it into a Plotly figure. No plotting or dataframe dependency lives
here, so it stays cheap and unit-testable.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

# Column names that almost always mean "time axis".
_TIME_NAMES = frozenset(
    {
        "period",
        "date",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        "time",
        "ts",
        "timestamp",
        "datetime",
        # Sequential index columns (e.g. forecast steps) also make a sensible
        # line-chart x-axis even though they are not temporal.
        "step",
        "index",
    }
)


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """A minimal, backend-agnostic description of a chart to render."""

    kind: str  # "line" | "bar" | "histogram"
    x: str
    y: str | None = None
    title: str = ""


def choose_chart(
    rows: list[dict] | None, *, title: str = "", max_bars: int = 30
) -> ChartSpec | None:
    """Pick a chart for these result rows, or ``None`` if nothing sensible fits.

    Rules, in order: time-like column + a measure → line; a category + a measure
    (bounded row count) → bar; a single numeric column over many rows → histogram.
    """
    if not rows or not isinstance(rows, list) or not isinstance(rows[0], dict):
        return None
    cols = list(rows[0].keys())
    if not cols:
        return None

    numeric = [c for c in cols if _is_numeric(rows, c)]
    categorical = [c for c in cols if c not in numeric]
    time_cols = [c for c in cols if c.lower() in _TIME_NAMES] or [
        c for c in categorical if _looks_temporal(rows, c)
    ]

    # Time series → line
    if time_cols and numeric:
        y = _first_not(numeric, time_cols)
        if y is not None:
            return ChartSpec("line", x=time_cols[0], y=y, title=title)

    # Category breakdown → bar (only when the number of bars is readable)
    if categorical and numeric and len(rows) <= max_bars:
        x = categorical[0]
        y = _first_not(numeric, [x])
        if y is not None:
            return ChartSpec("bar", x=x, y=y, title=title)

    # A single numeric column over many rows → distribution
    if len(numeric) == 1 and not categorical and len(rows) > 1:
        return ChartSpec("histogram", x=numeric[0], y=None, title=title)

    return None


def _is_numeric(rows: list[dict], col: str) -> bool:
    seen = False
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        seen = True
    return seen


def _looks_temporal(rows: list[dict], col: str) -> bool:
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        if isinstance(v, (_dt.date, _dt.datetime)):
            return True
        # ISO-ish date string: YYYY-MM... or YYYY/MM...
        return isinstance(v, str) and len(v) >= 8 and v[:4].isdigit() and v[4:5] in {"-", "/"}
    return False


def _first_not(candidates: list[str], exclude: list[str]) -> str | None:
    for c in candidates:
        if c not in exclude:
            return c
    return None
