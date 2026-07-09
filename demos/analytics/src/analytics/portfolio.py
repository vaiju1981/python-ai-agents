"""Portfolio / budget optimizer (generic lift of ATLAS optimizer/portfolio.py).

Two strategies, both column-agnostic:

* ``greedy_budget_frontier`` — sort candidate actions by a value/ROI objective
  and walk down, enforcing a per-group diversity cap, to trace the cumulative
  value vs number-of-changes efficient frontier.
* ``pareto_frontier`` — multi-objective selection under a budget. Uses NSGA-II
  (pymoo) when available, otherwise a weighted-sum sweep that returns the
  non-dominated set. Approved/``reserved`` actions are forced into every plan.
"""

from __future__ import annotations

from typing import Any

INTERACTION_WEIGHT = float(25.0)


def greedy_budget_frontier(
    recs: list[dict[str, Any]],
    value_axis: str = "value",
    max_changes: int = 24,
    per_group_cap: int = 1,
    group_col: str = "group",
    constraints: Any = None,
    reserved: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Greedy cumulative-value frontier with a per-group diversity cap."""
    reserved = reserved or set()
    eligible = [r for r in recs if r.get(value_axis, 0) > 0]
    eligible.sort(key=lambda r: float(r.get(value_axis, 0)), reverse=True)

    group_count: dict[Any, int] = {}
    selected: list[str] = []
    cum = 0.0
    frontier: list[dict[str, Any]] = []

    def _can_add(r: dict[str, Any]) -> bool:
        if constraints is not None and getattr(constraints, "first_violation", None):
            if constraints.first_violation(r):
                return False
        g = r.get(group_col)
        if g is not None and group_count.get(g, 0) >= per_group_cap:
            return False
        return True

    for r in recs:
        if r.get("id") in reserved:
            g = r.get(group_col)
            if g is not None:
                group_count[g] = group_count.get(g, 0) + 1
            selected.append(r["id"])
            cum += float(r.get(value_axis, 0))

    for r in eligible:
        if len(selected) >= max_changes:
            break
        if r.get("id") in selected:
            continue
        if not _can_add(r):
            continue
        g = r.get(group_col)
        if g is not None:
            group_count[g] = group_count.get(g, 0) + 1
        selected.append(r["id"])
        cum += float(r.get(value_axis, 0))
        frontier.append(
            {"nChanges": len(selected), "cumValue": round(cum, 4), "selected": list(selected)}
        )
    return frontier


def pareto_frontier(
    recs: list[dict[str, Any]],
    budget: int = 12,
    objectives: list[tuple[str, str]] | None = None,
    group_col: str = "group",
    per_group_cap: int = 1,
    reserved: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Multi-objective selection under a budget (maximize 'max', minimize 'min').

    Returns the non-dominated set of plans as ``{selected, objectives, score}``.
    """
    reserved = reserved or set()
    if objectives is None:
        objectives = [("value", "max")]
    valid = [r for r in recs if r.get("id") is not None]
    # Normalize each objective to [0, 1] for fair weighting.
    norms: dict[str, list[float]] = {}
    for key, _dir in objectives:
        vals = [float(r.get(key, 0)) for r in valid]
        lo, hi = (min(vals), max(vals)) if vals else (0, 0)
        rng = hi - lo or 1.0
        norms[key] = [(v - lo) / rng for v in vals]

    try:
        from pymoo.optimize import minimize  # type: ignore
        from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore
        from pymoo.core.problem import ElementwiseProblem  # type: ignore
        from pymoo.operators.crossover.pntx import TwoPointCrossover  # type: ignore
        from pymoo.operators.mutation.bitflip import BitflipMutation  # type: ignore
        from pymoo.operators.sampling.rnd import BinaryRandomSampling  # type: ignore
        from pymoo.termination import get_termination  # type: ignore

        n = len(valid)

        class _Prob(ElementwiseProblem):
            def __init__(self) -> None:
                super().__init__(n_var=n, n_obj=len(objectives), n_ieq_constr=1, vtype=bool)

            def _evaluate(self, x, out):  # type: ignore
                sel = [valid[i] for i in range(n) if x[i]]
                obj_vals = []
                for key, direction in objectives:
                    s = sum(float(r.get(key, 0)) for r in sel)
                    obj_vals.append(s if direction == "max" else -s)
                gc: dict[Any, int] = {}
                for r in sel:
                    g = r.get(group_col)
                    if g is not None:
                        gc[g] = gc.get(g, 0) + 1
                inter = sum(max(0, c - 1) for c in gc.values())
                out["F"] = [-(o - INTERACTION_WEIGHT * inter / max(1, len(sel))) for o in obj_vals]
                out["G"] = [sum(x) - budget]

        res = minimize(
            _Prob(),
            NSGA2(pop_size=80, crossover=TwoPointCrossover(), mutation=BitflipMutation(),
                  sampling=BinaryRandomSampling()),
            get_termination("n_gen", 150), seed=42, verbose=False,
        )
        plans: list[dict[str, Any]] = []
        for x in res.X:
            sel = [valid[i]["id"] for i in range(n) if x[i]]
            plans.append(_plan_dict(sel, valid, objectives, norms))
        return _non_dominated(plans, objectives, norms, valid)
    except Exception:
        # Weighted-sum sweep → non-dominated set (no pymoo dependency).
        plans: list[dict[str, Any]] = []
        steps = 5
        import itertools

        for weights in itertools.product(*[np_linspace() for _ in objectives]):
            w = [x / sum(weights) for x in weights]
            ranked = sorted(
                range(len(valid)),
                key=lambda i: sum(wi * norms[key][i] * (1 if d == "max" else -1)
                                   for (key, d), wi in zip(objectives, w)),
                reverse=True,
            )
            sel: list[str] = []
            gc: dict[Any, int] = {}
            for i in ranked:
                rid = valid[i]["id"]
                if rid in reserved:
                    sel.append(rid)
                    continue
                g = valid[i].get(group_col)
                if g is not None and gc.get(g, 0) >= per_group_cap:
                    continue
                if len([s for s in sel if s not in reserved]) >= budget:
                    break
                if g is not None:
                    gc[g] = gc.get(g, 0) + 1
                sel.append(rid)
            for rid in reserved:
                if rid not in sel:
                    sel.append(rid)
            plans.append(_plan_dict(sel, valid, objectives, norms))
        return _non_dominated(plans, objectives, norms, valid)


def _plan_dict(selected, valid, objectives, norms) -> dict[str, Any]:
    obj_vals = {key: round(sum(float(r.get(key, 0)) for r in valid if r["id"] in selected), 4)
                for key, _ in objectives}
    return {"selected": list(selected), "objectives": obj_vals}


def _non_dominated(plans, objectives, norms, valid) -> list[dict[str, Any]]:
    # Dedupe by selected set, then keep only Pareto-non-dominated plans.
    seen: dict[tuple, dict[str, Any]] = {}
    for p in plans:
        key = tuple(sorted(p["selected"]))
        if key not in seen:
            seen[key] = p
    unique = list(seen.values())
    keep = []
    for i, a in enumerate(unique):
        dominated = False
        for j, b in enumerate(unique):
            if i == j:
                continue
            if _dominates(b, a, objectives, norms, valid):
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return keep


def _dominates(a, b, objectives, norms, valid) -> bool:
    a_ids = set(a["selected"])
    b_ids = set(b["selected"])
    better = False
    for key, direction in objectives:
        av = sum(float(r.get(key, 0)) for r in valid if r["id"] in a_ids)
        bv = sum(float(r.get(key, 0)) for r in valid if r["id"] in b_ids)
        if direction == "max":
            if av < bv:
                return False
            if av > bv:
                better = True
        else:
            if av > bv:
                return False
            if av < bv:
                better = True
    return better


def np_linspace() -> list[float]:
    # Avoid a hard numpy dependency just for weight grid.
    return [0.0, 0.25, 0.5, 0.75, 1.0]
